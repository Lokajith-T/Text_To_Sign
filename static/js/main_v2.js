var label = document.getElementById("label");
var container = document.getElementById("container");


function getContainerDimensions() {
    const width = container.clientWidth || container.offsetWidth || 600;
    const height = container.clientHeight || container.offsetHeight || 500;
    return { width, height, aspect: width / height };
}

let dims = getContainerDimensions();
const scene = new THREE.Scene();
// scene.background = new THREE.Color(0x253238); // Make transparent to show CSS gradient
const camera = new THREE.PerspectiveCamera(65, dims.aspect, 0.1, 5000);
camera.position.set(25, -5, 75); // Zoomed in for the default model

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(dims.width, dims.height);
renderer.domElement.style.position = "absolute";
renderer.domElement.style.top = "0";
renderer.domElement.style.left = "0";
renderer.domElement.style.width = "100%";
renderer.domElement.style.height = "100%";
container.appendChild(renderer.domElement);


function updateCanvasSize() {
    let d = getContainerDimensions();
    camera.aspect = d.aspect;
    camera.updateProjectionMatrix();
    renderer.setSize(d.width, d.height);
}

window.addEventListener('resize', updateCanvasSize);
setTimeout(updateCanvasSize, 50);
setTimeout(updateCanvasSize, 300);


var data = {};
var wordList = [];
var wordidx = 0;
var frameidx = 0;

// --- 3D Rigged Model Implementation ---
// WARNING: Adjust these bone names to match your exact Blender rig!
// E.g. 'mixamorig:RightHandThumb1' or 'Hand_Thumb1_R'
const BONE_NAMES = {
    left: {
        upper_arm: 'mixamorig:LeftArm',
        forearm: 'mixamorig:LeftForeArm',
        wrist: 'mixamorig:LeftHand',
        thumb: ['mixamorig:LeftHandThumb1', 'mixamorig:LeftHandThumb2', 'mixamorig:LeftHandThumb3'],
        index: ['mixamorig:LeftHandIndex1', 'mixamorig:LeftHandIndex2', 'mixamorig:LeftHandIndex3'],
        middle: ['mixamorig:LeftHandMiddle1', 'mixamorig:LeftHandMiddle2', 'mixamorig:LeftHandMiddle3'],
        ring: ['mixamorig:LeftHandRing1', 'mixamorig:LeftHandRing2', 'mixamorig:LeftHandRing3'],
        pinky: ['mixamorig:LeftHandPinky1', 'mixamorig:LeftHandPinky2', 'mixamorig:LeftHandPinky3']
    },
    right: {
        upper_arm: 'mixamorig:RightArm',
        forearm: 'mixamorig:RightForeArm',
        wrist: 'mixamorig:RightHand',
        thumb: ['mixamorig:RightHandThumb1', 'mixamorig:RightHandThumb2', 'mixamorig:RightHandThumb3'],
        index: ['mixamorig:RightHandIndex1', 'mixamorig:RightHandIndex2', 'mixamorig:RightHandIndex3'],
        middle: ['mixamorig:RightHandMiddle1', 'mixamorig:RightHandMiddle2', 'mixamorig:RightHandMiddle3'],
        ring: ['mixamorig:RightHandRing1', 'mixamorig:RightHandRing2', 'mixamorig:RightHandRing3'],
        pinky: ['mixamorig:RightHandPinky1', 'mixamorig:RightHandPinky2', 'mixamorig:RightHandPinky3']
    }
};

// Scene Lighting
scene.add(new THREE.AmbientLight(0xffffff, 2.0)); // Increased intensity
const dirLight = new THREE.DirectionalLight(0xffffff, 2.0);
dirLight.position.set(25, -10, 20); // Moved light closer to model
scene.add(dirLight);

// Make camera look at the face/chest level
camera.lookAt(25, -15, 0);

let humanModel = null;
let rigBones = { left: {}, right: {} };

const loader = new THREE.GLTFLoader();
loader.load('static/model/ManModel.glb', function(gltf) {
    humanModel = gltf.scene;
    const box = new THREE.Box3().setFromObject(humanModel);
    console.log("Model Bounding Box:", box);

    // ManModel.glb specific scale and position
    humanModel.scale.set(20, 20, 20);
    humanModel.position.set(25, -50, 0);
    scene.add(humanModel);

    const sanitize = (name) => name.replace(/[^a-zA-Z0-9]/g, '').toLowerCase();

    const findBones = (boneMap) => {
        let result = {};
        humanModel.traverse((child) => {
            let childSan = sanitize(child.name);
            for (let key in boneMap) {
                if (Array.isArray(boneMap[key])) {
                    for (let n of boneMap[key]) {
                        if (childSan.includes(sanitize(n))) {
                            if (!result[key]) result[key] = [];
                            result[key].push(child);
                            break;
                        }
                    }
                } else if (childSan.includes(sanitize(boneMap[key]))) {
                    result[key] = child;
                }
            }
        });
        return result;
    };
    rigBones.left = findBones(BONE_NAMES.left);
    rigBones.right = findBones(BONE_NAMES.right);
    window.humanModel = humanModel;
    window.rigBones = rigBones;

    console.log("3D Model Loaded.");
    console.log("Left wrist found:", !!rigBones.left.wrist, "Right wrist found:", !!rigBones.right.wrist);
    if (!rigBones.right.wrist) {
        let names = [];
        humanModel.traverse(c => { if (c.name.includes('hand')) names.push(c.name); });
        console.log("Could not find right wrist! Nodes with 'hand' in name:", names);
    }

    // Add debugAnimateHandCount here to ensure it's initialized globally
    window.debugAnimateHandCount = 0;
}, undefined, function (error) {
    console.error("Error loading 3D model:", error);
});

function orientBone(bone, p1, p2, globalUp, zScale = 1.0) {
    if (!bone || !p1 || !p2) return;
    const v1 = new THREE.Vector3(p1.Coordinates[0], p1.Coordinates[1] * -1, p1.Coordinates[2] * -zScale);
    const v2 = new THREE.Vector3(p2.Coordinates[0], p2.Coordinates[1] * -1, p2.Coordinates[2] * -zScale);
    const targetDir = new THREE.Vector3().subVectors(v2, v1).normalize();

    let targetUp = globalUp ? globalUp.clone() : new THREE.Vector3(0, 0, 1);

    if (bone.parent) {
        bone.parent.updateWorldMatrix(true, false);
        const parentInverse = new THREE.Matrix4().copy(bone.parent.matrixWorld).invert();
        targetDir.transformDirection(parentInverse);
        targetUp.transformDirection(parentInverse);
    }

    let defaultDir = new THREE.Vector3(0, 1, 0); // fallback
    if (bone.children.length > 0) {
        // Find the first child that has an actual physical offset
        for (let child of bone.children) {
            if (child.position.lengthSq() > 0.0001) {
                defaultDir.copy(child.position).normalize();
                break;
            }
        }
    }

    // 1. Point the bone (shortest path)
    const pointQuat = new THREE.Quaternion().setFromUnitVectors(defaultDir, targetDir);

    // 2. Fix the twist (roll) if globalUp is provided
    if (globalUp) {
        // Find what the local up vector (0, 0, 1) currently points to after pointQuat
        let currentUp = new THREE.Vector3(0, 0, 1).applyQuaternion(pointQuat);
        
        // Project both the current up and the target up onto the plane perpendicular to targetDir
        let projectedCurrentUp = currentUp.clone().projectOnPlane(targetDir).normalize();
        let projectedTargetUp = targetUp.clone().projectOnPlane(targetDir).normalize();

        if (projectedCurrentUp.lengthSq() > 0.001 && projectedTargetUp.lengthSq() > 0.001) {
            // Find the rotation strictly around targetDir
            let twistQuat = new THREE.Quaternion().setFromUnitVectors(projectedCurrentUp, projectedTargetUp);
            pointQuat.premultiply(twistQuat);
        }
    }

    bone.quaternion.copy(pointQuat);
}

function animateHand(coords, bones, poseCoords, isLeft) {
    if (window.debugAnimateHandCount < 5) {
        console.log("DEBUG animateHand:", coords ? coords.length : 'null', "coords, wrist exists:", !!bones.wrist);
        if (!bones.wrist) console.log("Missing wrist in bones:", Object.keys(bones));
        window.debugAnimateHandCount++;
    }

    if (!coords || coords.length === 0 || !bones.wrist) return;

    // Use Pose Coordinates if available, otherwise fallback to Fake IK
    if (bones.upper_arm && poseCoords && poseCoords.length > 0) {
        const getCoord = (idx) => {
            const p = poseCoords.find(c => c["Joint Index"] === idx);
            if (!p) return null;
            return p.Coordinates;
        };

        const sIdx = isLeft ? 11 : 12; // Shoulder
        const eIdx = isLeft ? 13 : 14; // Elbow
        const wIdx = isLeft ? 15 : 16; // Wrist

        const cShoulder = getCoord(sIdx);
        const cElbow = getCoord(eIdx);
        const cWrist = getCoord(wIdx);

        if (cShoulder && cElbow && cWrist) {
            let armUp = new THREE.Vector3(0, 0, -1);
            let armZScale = 0.2; // Flatten Z to reduce extreme foreshortening
            orientBone(bones.upper_arm, {Coordinates: cShoulder}, {Coordinates: cElbow}, armUp, armZScale);
            if (bones.forearm) {
                orientBone(bones.forearm, {Coordinates: cElbow}, {Coordinates: cWrist}, armUp, armZScale);
            }
        }
    } else if (bones.upper_arm) {
        // --- FALLBACK TO FAKE IK ---
        const scale = 50;
        const vWristWorld = new THREE.Vector3(
            (coords[0].Coordinates[0] - 0.5) * scale + 25,
            (coords[0].Coordinates[1] - 0.5) * -scale - 20,
            (coords[0].Coordinates[2] * -scale) + 15
        );

        const vShoulder = new THREE.Vector3();
        bones.upper_arm.getWorldPosition(vShoulder);

        const vMidpoint = new THREE.Vector3().addVectors(vShoulder, vWristWorld).multiplyScalar(0.5);
        const vElbowTarget = vMidpoint.clone();
        vElbowTarget.y -= 15; 
        vElbowTarget.z -= 5;  

        const targetDirUpper = new THREE.Vector3().subVectors(vElbowTarget, vShoulder).normalize();
        if (bones.upper_arm.parent) {
            const parentInverse = new THREE.Matrix4().copy(bones.upper_arm.parent.matrixWorld).invert();
            targetDirUpper.transformDirection(parentInverse);
        }

        let defaultDirUpper = new THREE.Vector3(0, 1, 0);
        if (bones.forearm) {
            defaultDirUpper.copy(bones.forearm.position).normalize();
        }
        bones.upper_arm.quaternion.setFromUnitVectors(defaultDirUpper, targetDirUpper);

        if (bones.forearm) {
            bones.upper_arm.updateWorldMatrix(true, false);
            const vActualElbow = new THREE.Vector3();
            bones.forearm.getWorldPosition(vActualElbow);

            const targetDirLower = new THREE.Vector3().subVectors(vWristWorld, vActualElbow).normalize();
            if (bones.forearm.parent) {
                const parentInverse = new THREE.Matrix4().copy(bones.forearm.parent.matrixWorld).invert();
                targetDirLower.transformDirection(parentInverse);
            }

            let defaultDirLower = new THREE.Vector3(0, 1, 0);
            if (bones.wrist) {
                defaultDirLower.copy(bones.wrist.position).normalize();
            }
            bones.forearm.quaternion.setFromUnitVectors(defaultDirLower, targetDirLower);
        }
    }

    // Calculate Palm Normal for consistent finger roll
    let palmNormal = null;
    if (coords[0] && coords[5] && coords[17]) {
        const w = new THREE.Vector3(coords[0].Coordinates[0], coords[0].Coordinates[1] * -1, coords[0].Coordinates[2] * -1);
        const i = new THREE.Vector3(coords[5].Coordinates[0], coords[5].Coordinates[1] * -1, coords[5].Coordinates[2] * -1);
        const p = new THREE.Vector3(coords[17].Coordinates[0], coords[17].Coordinates[1] * -1, coords[17].Coordinates[2] * -1);
        const vA = new THREE.Vector3().subVectors(i, w);
        const vB = new THREE.Vector3().subVectors(p, w);
        // Ensure palmNormal points to the BACK of the hand for both left and right hands
        if (isLeft) {
            palmNormal = new THREE.Vector3().crossVectors(vA, vB).normalize();
        } else {
            palmNormal = new THREE.Vector3().crossVectors(vB, vA).normalize();
        }
    }

    // Thumb
    if (bones.thumb) {
        orientBone(bones.thumb[0], coords[1], coords[2], palmNormal);
        orientBone(bones.thumb[1], coords[2], coords[3], palmNormal);
        orientBone(bones.thumb[2], coords[3], coords[4], palmNormal);
    }
    // Index
    if (bones.index) {
        orientBone(bones.index[0], coords[5], coords[6], palmNormal);
        orientBone(bones.index[1], coords[6], coords[7], palmNormal);
        orientBone(bones.index[2], coords[7], coords[8], palmNormal);
    }
    // Middle
    if (bones.middle) {
        orientBone(bones.middle[0], coords[9], coords[10], palmNormal);
        orientBone(bones.middle[1], coords[10], coords[11], palmNormal);
        orientBone(bones.middle[2], coords[11], coords[12], palmNormal);
    }
    // Ring
    if (bones.ring) {
        orientBone(bones.ring[0], coords[13], coords[14], palmNormal);
        orientBone(bones.ring[1], coords[14], coords[15], palmNormal);
        orientBone(bones.ring[2], coords[15], coords[16], palmNormal);
    }
    // Pinky
    if (bones.pinky) {
        orientBone(bones.pinky[0], coords[17], coords[18], palmNormal);
        orientBone(bones.pinky[1], coords[18], coords[19], palmNormal);
        orientBone(bones.pinky[2], coords[19], coords[20], palmNormal);
    }

    // Rotate wrist to point from wrist (0) to middle finger base (9)
    orientBone(bones.wrist, coords[0], coords[9], palmNormal);
}

function getHandCoordinates(frameData) {
    let left = (frameData['Left Hand Coordinates'] || []).map(j => ({ ...j, Coordinates: [...j.Coordinates] }));
    let right = (frameData['Right Hand Coordinates'] || []).map(j => ({ ...j, Coordinates: [...j.Coordinates] }));
    let pose = (frameData['Pose Coordinates'] || []).map(j => ({ ...j, Coordinates: [...j.Coordinates] }));
    if (left.length > 21) {
        const extra = left.splice(21);
        right.push(...extra);
    } else if (right.length > 21) {
        const extra = right.splice(21);
        left.push(...extra);
    }
    return { left, right, pose };
}

let clock = new THREE.Clock();
let delta = 0;
let fps = 30; // 30 FPS fast & fluid sign animation
let interval = 1 / fps;
let pauseCounter = 0;
const PAUSE_FRAMES_BETWEEN_WORDS = 4; // Short ~0.1s natural pause

function render() {
    requestAnimationFrame(render);
    let dt = clock.getDelta();
    if (dt > 0.1) dt = 0.1;
    delta += dt;

    if (delta > interval) {
        delta = delta % interval;

        if (wordList.length > 0 && wordidx < wordList.length) {
            let currentWord = wordList[wordidx];
            label.innerHTML = currentWord.toUpperCase();

            if (pauseCounter > 0) {
                pauseCounter--;
            } else if (data[currentWord] && data[currentWord].length > 0 && data[currentWord][frameidx]) {
                const { left, right, pose } = getHandCoordinates(data[currentWord][frameidx]);

                // Animate the 3D model if it's loaded
                if (humanModel) {
                    animateHand(left, rigBones.left, pose, true);
                    animateHand(right, rigBones.right, pose, false);
                }

                frameidx++;
                if (frameidx >= data[currentWord].length) {
                    frameidx = 0;
                    wordidx++;
                    pauseCounter = PAUSE_FRAMES_BETWEEN_WORDS;
                }
            } else {
                console.log("Word landmark data missing or empty:", currentWord);
                frameidx = 0;
                wordidx++;
                pauseCounter = PAUSE_FRAMES_BETWEEN_WORDS;
            }
        } else if (wordList.length > 0) {
            // Loop animation sequence continuously for smooth viewing
            wordidx = 0;
            frameidx = 0;
        } else {
            label.innerHTML = "N/A";
        }
        renderer.render(scene, camera);
    }
}




// Start rendering Three.js scene immediately on page load
render();

function fetchAndAnimate(text) {
    fetch('/get_sign_data', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text })
    })
        .then(res => res.json())
        .then(res => {
            data = res.data || {};
            wordList = res.words || [];
            wordidx = 0;
            frameidx = 0;
            console.log("Sign data loaded for words:", wordList);
        })
        .catch(err => console.error("Error fetching sign data:", err));
}

var textForm = document.getElementById("inputForm");
if (textForm) {
    textForm.addEventListener("submit", function (e) {
        e.preventDefault();
        var message = document.getElementById("message").value;
        fetchAndAnimate(message);
    });
}

// --- Tab Navigation Logic ---
window.switchTab = function (tabId) {
    // Remove active class from all tabs and contents
    document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

    // Add active class to the selected tab and content
    document.getElementById(tabId).classList.add('active');

    // Find the button that called this (based on onclick attribute) and make it active
    const buttons = document.querySelectorAll('.tab-button');
    for (let btn of buttons) {
        if (btn.getAttribute('onclick').includes(tabId)) {
            btn.classList.add('active');
            break;
        }
    }
};
