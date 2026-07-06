/*==================================================
        DEEPFAKE DETECTION CONSOLE
                PART 1
==================================================*/

/*==========================
GET HTML ELEMENTS
==========================*/

const dropArea = document.getElementById("dropArea");
const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const uploadContent = document.getElementById("uploadContent");
const removeImage = document.getElementById("removeImage");

const chooseAnother =
document.getElementById("chooseAnother");

const runDetection =
document.getElementById("runDetection");

const loadingSection =
document.getElementById("loadingSection");

const vitSection =
document.getElementById("vitSection");

const prediction =
document.getElementById("prediction");

const confidence =
document.getElementById("confidence");

const confidenceFill =
document.getElementById("confidenceFill");



const statusText =
document.getElementById("statusText");

const inferenceTime =
document.getElementById("inferenceTime");





const patchImage =
document.getElementById("patchImage");

const heatmapImage =
document.getElementById("heatmapImage");

const embeddingImage =
document.getElementById("embeddingImage");

const realBar =
document.getElementById("realBar");

const fakeBar =
document.getElementById("fakeBar");



/*==========================
OPEN FILE PICKER
==========================*/

dropArea.addEventListener("click", () => {

    imageInput.click();

});

chooseAnother.addEventListener("click", () => {

    imageInput.click();

});


/*==========================
IMAGE SELECT
==========================*/

imageInput.addEventListener("change", () => {

    if(imageInput.files.length > 0){

        loadImage(imageInput.files[0]);

    }

});


/*==========================
LOAD IMAGE
==========================*/

function loadImage(file){

    const reader = new FileReader();

    reader.onload = function(e){

        previewImage.src = e.target.result;

        patchImage.style.backgroundImage = `url('${e.target.result}')`;

        document.querySelectorAll('.token').forEach(el => el.style.backgroundImage = `url('${e.target.result}')`);
        
        document.querySelectorAll('.heatmap div').forEach(el => el.style.backgroundImage = `url('${e.target.result}')`);

        previewImage.style.display = "block";

        uploadContent.style.display = "none";

        removeImage.style.display = "flex";

        statusText.innerHTML =
        "Image uploaded successfully.";

    };

    reader.readAsDataURL(file);

}


/*==========================
REMOVE IMAGE
==========================*/

removeImage.addEventListener("click",(e)=>{

    e.stopPropagation();

    imageInput.value = "";

    previewImage.src = "";

    patchImage.style.backgroundImage = "none";

    document.querySelectorAll('.token').forEach(el => el.style.backgroundImage = "none");
    
    document.querySelectorAll('.heatmap div').forEach(el => el.style.backgroundImage = "none");

    previewImage.style.display = "none";

    uploadContent.style.display = "flex";

    removeImage.style.display = "none";

    resetDashboard();

});


/*==========================
DRAG & DROP
==========================*/

dropArea.addEventListener("dragover",(e)=>{

    e.preventDefault();

    dropArea.classList.add("drag");

});

dropArea.addEventListener("dragleave",()=>{

    dropArea.classList.remove("drag");

});

dropArea.addEventListener("drop",(e)=>{

    e.preventDefault();

    dropArea.classList.remove("drag");

    const file = e.dataTransfer.files[0];

    if(file){

        imageInput.files = e.dataTransfer.files;

        loadImage(file);

    }

});


/*==========================
RUN DETECTION
==========================*/

runDetection.addEventListener("click",()=>{

    if(imageInput.files.length===0){

        alert("Please upload an image first.");

        return;

    }

    detectImage();

});
/*==================================================
            PART 2
        FLASK BACKEND CONNECTION
==================================================*/

async function detectImage(){

    loadingSection.style.display = "block";

    vitSection.style.display = "none";

    prediction.innerHTML = "PROCESSING...";

    prediction.className = "waiting";

    statusText.innerHTML =
    "Vision Transformer is analysing the image...";

    runDetection.disabled = true;

    runDetection.innerHTML =
    "ANALYSING...";

    const formData = new FormData();

    formData.append(
        "image",
        imageInput.files[0]
    );

    try{

        const response = await fetch("/predict",{

            method:"POST",

            body:formData

        });

        if(!response.ok){

            throw new Error("Server Error");

        }

        const data = await response.json();

        updateDashboard(data);

    }

    catch(error){

        console.error(error);

        alert("Unable to connect to Flask Backend.");

        runDetection.disabled = false;

        runDetection.innerHTML =
        "RUN DETECTION";

    }

}


/*==================================================
        UPDATE DASHBOARD
==================================================*/

function updateDashboard(data){

    loadingSection.style.display = "none";

    vitSection.style.display = "block";

    runDetection.disabled = false;

    runDetection.innerHTML =
    "RUN DETECTION";

    const result = data.prediction;

    const conf = Number(data.confidence);

    let real;

    let fake;

    if(result==="REAL"){

        real = conf;

        fake = 100-conf;

        prediction.innerHTML="REAL";

        statusText.innerHTML=
        "Authentic image detected.";

    }

    else{

        fake = conf;

        real = 100-conf;

        prediction.innerHTML="FAKE";

        prediction.className="fake";

        statusText.innerHTML=
        "Deepfake detected.";

    }

    confidence.innerHTML =
    conf.toFixed(2)+"%";


}
/*==================================================
                PART 3
        ANIMATIONS & DASHBOARD
==================================================*/




/*==========================
ANIMATE ANALYSIS
==========================*/



/*==========================
PROGRESS ANIMATION
==========================*/

function animateProgress(bar,label,target){

    bar.style.width = target + "%";

    let value = 0;

    label.innerHTML = "0%";

    const timer = setInterval(()=>{

        value++;

        label.innerHTML = value + "%";

        if(value >= target){

            clearInterval(timer);

        }

    },10);

}


/*==========================
RESET DASHBOARD
==========================*/

function resetDashboard(){

    prediction.innerHTML = "WAITING...";

    prediction.className = "waiting";

    confidence.innerHTML = "--";
    
    if (confidenceFill) {
        confidenceFill.style.width = "0%";
    }
    
    if (inferenceTime) {
        inferenceTime.innerHTML = "--";
        inferenceTime.style.color = "#9BA7C8";
    }
    
    if (heatmapImage) {
        heatmapImage.style.display = "none";
        heatmapImage.src = "";
    }
    if (embeddingImage) {
        embeddingImage.style.display = "none";
        embeddingImage.src = "";
    }
    if (realBar) realBar.style.width = "0%";
    if (fakeBar) fakeBar.style.width = "0%";





    statusText.innerHTML =
    "Awaiting Image...";




    loadingSection.style.display = "none";

    vitSection.style.display = "none";

    runDetection.disabled = false;

    runDetection.innerHTML =
    "RUN DETECTION";

}


/*==========================
UTILITY
==========================*/

function randomNumber(min,max){

    return Math.floor(

        Math.random() * (max-min+1)

    ) + min;

}


/*==========================
INITIALIZE
==========================*/

window.addEventListener("load",()=>{

    resetDashboard();

});
/*==================================================
                PART 4
        FINAL EFFECTS & UTILITIES
==================================================*/

/*==========================
NUMBER COUNT ANIMATION
==========================*/

function animateCounter(element, target, suffix = "%"){

    let current = 0;
    const finalTarget = Number(target);
    const intTarget = Math.floor(finalTarget);

    const timer = setInterval(()=>{

        current += 2; // speed up animation

        if(current >= intTarget){

            clearInterval(timer);
            element.innerHTML = finalTarget.toFixed(1) + suffix; // Show exactly 1 decimal place, e.g. 97.8%

        } else {
            element.innerHTML = current + suffix;
        }

    },15);

}

/*==========================
UPDATE SCORES
==========================*/

function updateScores(real,fake,conf){

    animateCounter(confidence,conf);
    
    if (confidenceFill) {
        confidenceFill.style.width = conf + "%";
    }

}

/*==========================
SHOW NOTIFICATION
==========================*/

function showNotification(message,color="#4FCBFF"){

    statusText.innerHTML = message;

    statusText.style.color = color;

}

/*==========================
SUCCESS EFFECT
==========================*/

function successAnimation(){

    prediction.style.transform = "scale(1.15)";

    setTimeout(()=>{

        prediction.style.transform = "scale(1)";

    },300);

}

/*==========================
MODIFY DASHBOARD FUNCTION
==========================*/

const oldUpdateDashboard = updateDashboard;

updateDashboard = function(data){

    oldUpdateDashboard(data);

    const conf = Number(data.confidence);

    let real;
    let fake;

    if(data.prediction === "REAL"){

        real = conf;

        fake = 100-conf;

        showNotification(
            "Authentic image detected.",
            "#33ff99"
        );

    }
    else{

        fake = conf;

        real = 100-conf;

        showNotification(
            "Deepfake image detected.",
            "#ff5577"
        );

    }

    updateScores(real,fake,conf);
    
    if (realBar) realBar.style.width = real + "%";
    if (fakeBar) fakeBar.style.width = fake + "%";
    if (data.heatmap && heatmapImage) {
        heatmapImage.src = "data:image/jpeg;base64," + data.heatmap;
        heatmapImage.style.display = "block";
    }
    
    if (data.embedding && embeddingImage) {
        embeddingImage.src = "data:image/jpeg;base64," + data.embedding;
        embeddingImage.style.display = "block";
    }
    
    if (data.inference_time && inferenceTime) {
        inferenceTime.innerHTML = data.inference_time + "s";
        inferenceTime.style.color = "#4FCBFF";
    }

    successAnimation();

}

/*==========================
CONSOLE MESSAGE
==========================*/

console.log(
"%cDeepFake Detection Console Loaded",
"color:#4FCBFF;font-size:18px;font-weight:bold;"
);