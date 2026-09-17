"""Streamlit controller/state and failure rendering checks, with mocked inference."""
import copy,io,json,sys
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def png(color):
    b=io.BytesIO();Image.new("RGB",(64,64),color).save(b,format="PNG");return b.getvalue()
class Upload:
    name="example.png"
    def __init__(self,data):self.data=data
    def getvalue(self):return self.data
def main():
    valid={"status":"lemon_leaf_analysis","validator":{"lemon_score":.98,"accepted_as_lemon_leaf":True},
           "predicted_class":"Healthy leaf","predicted_class_index":5,"confidence":.8,
           "top_3":[{"class":"Healthy leaf","class_index":5,"probability":.8},
                    {"class":"Greening","class_index":4,"probability":.15},{"class":"Black spot","class_index":1,"probability":.05}],
           "probabilities":[0,.05,0,0,.15,.8],"legacy_uncertain":False,"warnings":[],
           "gradcam":{"status":"available"},"error":None}
    cases=[
      ("technical",{"status":"technical_error","error":{"detail":"Simulated corrupt file"}},"We couldn't process"),
      ("validator_failure",{"status":"analysis_error","validator":None,"error":{"stage":"validator","detail":"Simulated unavailable model"}},"Input recognition is temporarily unavailable"),
      ("disease_failure",{"status":"analysis_error","validator":{"accepted_as_lemon_leaf":True},"error":{"stage":"disease_classifier","detail":"Simulated failure"}},"We couldn't complete disease analysis")]
    outcomes=[]
    import utils.prediction
    for name,result,message in cases:
        result.setdefault("gradcam", {"status":"not_run"})
        with patch("streamlit.file_uploader",return_value=Upload(png("green"))),patch.object(utils.prediction,"analyze_image",return_value=copy.deepcopy(result)):
            app=AppTest.from_file(str(ROOT/"streamlit_app.py"),default_timeout=30).run()
            next(b for b in app.button if b.label=="Analyze Leaf").click().run()
            assert not app.exception
            assert any(message in e.value for e in app.error)
            assert not any("Top 3 predictions" in m.value for m in app.markdown)
        outcomes.append({"case":name,"passed":True})
    upload=Upload(png("green"))
    def backend(contents,output):
        Image.new("RGB",(64,64),"yellow").save(output)
        return copy.deepcopy(valid)
    with patch("streamlit.file_uploader",return_value=upload),patch.object(utils.prediction,"analyze_image",side_effect=backend) as inference:
        app=AppTest.from_file(str(ROOT/"streamlit_app.py"),default_timeout=30).run()
        assert inference.call_count==0
        next(b for b in app.button if b.label=="Analyze Leaf").click().run()
        assert not app.exception and inference.call_count==1
        app.toggle[0].set_value(False).run()
        app.run()
        assert inference.call_count==1,"UI reruns must not repeat inference"
        assert next(b for b in app.button if b.label=="Analyze Leaf").disabled
        upload.data=png("blue")
        app.run()
        assert app.session_state["analysis"] is None
        assert not next(b for b in app.button if b.label=="Analyze Leaf").disabled
        assert inference.call_count==1
    outcomes.append({"case":"no_auto_inference_cached_reruns_new_upload_invalidation","passed":True})
    for name,uncertain in [("gradcam_failure",False),("provisional_uncertainty",True)]:
        result=copy.deepcopy(valid)
        result["gradcam"]={"status":"unavailable"}
        result["legacy_uncertain"]=uncertain
        with patch("streamlit.file_uploader",return_value=Upload(png("green"))),patch.object(utils.prediction,"analyze_image",return_value=result):
            app=AppTest.from_file(str(ROOT/"streamlit_app.py"),default_timeout=30).run()
            next(b for b in app.button if b.label=="Analyze Leaf").click().run()
            assert not app.exception
            assert any("Top 3 predictions" in m.value for m in app.markdown)
            assert any("Visual explanation unavailable" in m.value for m in app.info)
            if uncertain:assert any("Uncertain result" in m.value for m in app.warning)
        outcomes.append({"case":name,"passed":True})
    (ROOT/"reports/streamlit_ui/controller_smoke_results.json").write_text(json.dumps(outcomes,indent=2)+"\n")
    print("ALL CONTROLLER/ERROR-STATE CHECKS PASSED")
if __name__=="__main__":main()

