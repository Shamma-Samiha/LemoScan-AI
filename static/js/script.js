const imageInput = document.getElementById("image");
const previewImage = document.getElementById("preview-image");

if (imageInput) {
    imageInput.addEventListener("change", function () {
        const file = this.files[0];

        if (file) {
            const reader = new FileReader();

            reader.onload = function (event) {
                previewImage.setAttribute("src", event.target.result);
                previewImage.style.display = "block";
            };

            reader.readAsDataURL(file);
        }
    });
}