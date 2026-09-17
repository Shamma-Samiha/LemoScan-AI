"""Local Edge DevTools smoke harness; uses the installed browser, no extra package."""
import asyncio,base64,json,urllib.request,sys
from pathlib import Path
import websockets
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/streamlit_ui"
class Browser:
    async def __aenter__(self):
        tabs=json.load(urllib.request.urlopen("http://127.0.0.1:9229/json"))
        target=next(t for t in tabs if t["type"]=="page" and "8510" in t["url"])
        self.ws=await websockets.connect(target["webSocketDebuggerUrl"],max_size=30_000_000)
        self.counter=0
        return self
    async def __aexit__(self,*args):
        await self.ws.close()
    async def call(self,method,params=None):
        self.counter+=1
        ident=self.counter
        await self.ws.send(json.dumps({"id":ident,"method":method,"params":params or {}}))
        while True:
            response=json.loads(await self.ws.recv())
            if response.get("id")==ident:
                if "error" in response: raise RuntimeError(response["error"])
                return response.get("result",{})
    async def evaluate(self,code):
        response=await self.call("Runtime.evaluate",{"expression":code,"returnByValue":True,"awaitPromise":True})
        if "exceptionDetails" in response: raise RuntimeError(response["exceptionDetails"])
        return response["result"].get("value")
    async def screenshot(self,name):
        value=await self.call("Page.captureScreenshot",{"format":"png","captureBeyondViewport":False})
        (OUT/name).write_bytes(base64.b64decode(value["data"]))
    async def wait_text(self,text,timeout=90):
        for _ in range(timeout*2):
            content=await self.evaluate("document.body?.innerText || ''")
            if text in content:return content
            await asyncio.sleep(.5)
        raise AssertionError(f"Did not see {text!r}; last page: {content[-2000:]}")
    async def upload(self,path):
        doc=await self.call("DOM.getDocument")
        node=await self.call("DOM.querySelector",{"nodeId":doc["root"]["nodeId"],"selector":"input[type=file]"})
        assert node["nodeId"]
        await self.call("DOM.setFileInputFiles",{"nodeId":node["nodeId"],"files":[str(Path(path).resolve())]})
    async def click(self,label):
        result=await self.evaluate("(()=>{const b=[...document.querySelectorAll('button')].find(x=>x.innerText.trim()==="+json.dumps(label)+");if(!b||b.disabled)return false;b.click();return true})()")
        assert result,f"Button unavailable: {label}"
async def main():
    async with Browser() as browser:
        await browser.call("Emulation.setDeviceMetricsOverride",{"width":1440,"height":1080,"deviceScaleFactor":1,"mobile":False})
        await browser.wait_text("Upload a lemon leaf image",30)
        print(await browser.evaluate("JSON.stringify({text:document.body.innerText,inputs:[...document.querySelectorAll('input')].map(x=>({type:x.type,label:x.getAttribute('aria-label')})),buttons:[...document.querySelectorAll('button')].map(x=>x.innerText)})"))
        await browser.screenshot("landing_desktop.png")



async def smoke():
    import pandas as pd
    from PIL import Image
    frames=pd.read_csv(ROOT/"reports/final_split/final_split_manifest.csv")
    leaf=ROOT/frames[frames.new_split.eq("train")].sort_values("filepath").iloc[0].filepath
    negative=pd.read_csv(ROOT/"reports/leaf_validator/validator_dataset_manifest.csv")
    bean=ROOT/negative[(negative.source=="beans")&(negative.split=="train")].sort_values("filepath").iloc[0].filepath
    obj=ROOT/negative[(negative.source=="caltech101")&(negative.split=="train")].sort_values("filepath").iloc[0].filepath
    green=OUT/"ui_solid_green.png"
    Image.new("RGB",(224,224),(0,255,0)).save(green)
    corrupt=OUT/"ui_corrupt.png"
    corrupt.write_bytes(b"not a valid image")
    results=[]
    async with Browser() as browser:
        await browser.call("Page.reload",{"ignoreCache":True})
        await browser.call("Emulation.setDeviceMetricsOverride",{"width":1440,"height":1080,"deviceScaleFactor":1,"mobile":False})
        await browser.wait_text("Upload a lemon leaf image",30)
        await asyncio.sleep(1)
        await browser.screenshot("landing_desktop.png")
        metric_cards=await browser.evaluate("[...document.querySelectorAll('.sidebar-metric strong')].map(x=>({text:x.innerText,client:x.clientWidth,scroll:x.scrollWidth}))")
        assert [m["text"] for m in metric_cards]==["95.44%","95.51%"]
        assert all(m["scroll"]<=m["client"] for m in metric_cards),metric_cards
        results.append({"case":"sidebar_metrics_full_values_no_clipping","passed":True,"metrics":metric_cards})
        for name,path,expected in [("valid_lemon",leaf,"Your leaf analysis"),("solid_green",green,"Input not recognized as a lemon leaf"),
                                    ("bean",bean,"Input not recognized as a lemon leaf"),("object",obj,"Input not recognized as a lemon leaf"),
                                    ("corrupt",corrupt,"We couldn't process this image.")]:
            await browser.upload(path)
            await browser.wait_text(path.name,30)
            for _ in range(60):
                ready=await browser.evaluate("[...document.querySelectorAll('button')].some(b=>b.innerText.trim()==='Analyze Leaf'&&!b.disabled)")
                if ready:break
                await asyncio.sleep(.5)
            assert ready
            before=await browser.evaluate("document.body.innerText")
            assert "Your leaf analysis" not in before,"Stale disease result remained after changing upload."
            await browser.click("Analyze Leaf")
            text=await browser.wait_text(expected,120)
            assert "Traceback" not in text
            if name=="valid_lemon":
                await browser.wait_text("Show visual explanation",30)
                text=await browser.evaluate("document.body.innerText")
                assert "Top 3 predictions" in text and "Disease-model confidence" in text
                assert await browser.evaluate("document.querySelectorAll('img[alt=\"Grad-CAM explanation of the predicted class\"]').length")==1
                await browser.evaluate("[...document.querySelectorAll('h2')].find(x=>x.innerText.includes('Your leaf analysis')).scrollIntoView()")
                # Native toggle triggers a genuine Streamlit rerun of the same upload.
                await browser.evaluate("document.querySelector('input[role=switch],input[type=checkbox]').click()")
                for _ in range(30):
                    if await browser.evaluate("document.querySelectorAll('img[alt=\"Grad-CAM explanation of the predicted class\"]').length")==0:break
                    await asyncio.sleep(.2)
                repeated=await browser.evaluate("document.body.innerText")
                assert "Your leaf analysis" in repeated and "Disease-model confidence" in repeated
                assert await browser.evaluate("[...document.querySelectorAll('button')].find(x=>x.innerText.trim()==='Analyze Leaf').disabled")
                await browser.evaluate("document.querySelector('input[role=switch],input[type=checkbox]').click()")
                for _ in range(30):
                    if await browser.evaluate("document.querySelectorAll('img[alt=\"Grad-CAM explanation of the predicted class\"]').length")==1:break
                    await asyncio.sleep(.2)
                results.append({"case":"same_upload_repeated_rerun","passed":True,"prediction_preserved":True})
            else:
                assert "Top 3 predictions" not in text and "Disease-model confidence" not in text
                assert await browser.evaluate("document.querySelectorAll('img[alt=\"Grad-CAM explanation of the predicted class\"]').length")==0
                await browser.evaluate("document.querySelector('[data-testid=stAlert]').scrollIntoView()")
            await asyncio.sleep(.3)
            await browser.screenshot(name+".png")
            results.append({"case":name,"passed":True,"expected_state":expected,"no_visible_traceback":True})
            print("PASS browser upload:",name,flush=True)
        await browser.call("Emulation.setDeviceMetricsOverride",{"width":390,"height":844,"deviceScaleFactor":1,"mobile":True})
        await browser.call("Page.reload",{"ignoreCache":True})
        await browser.wait_text("Upload a lemon leaf image",30)
        await asyncio.sleep(1)
        overflow=await browser.evaluate("document.documentElement.scrollWidth>window.innerWidth")
        assert not overflow,"Mobile horizontal overflow"
        await browser.screenshot("landing_mobile.png")
        results.append({"case":"mobile_390px","passed":True,"horizontal_overflow":False})
    (OUT/"browser_smoke_results.json").write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    print("ALL BROWSER UPLOAD CHECKS PASSED",flush=True)


if __name__=="__main__":asyncio.run(smoke())
