let x_vals = [];
let y_vals = [];
let a, b, c, d;
let dragging = false;
let predicting = true;
const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

function setup() {
    createCanvas(600, 600);
    userStartAudio();
    initializeVariables();
    document.getElementById('generateButton').addEventListener('click', trainModel);
    document.getElementById('stopButton').addEventListener('click', stopPredictions);
    document.getElementById('resetBtn').addEventListener('click', resetModel); // Make sure you have a reset button in your HTML
}

function initializeVariables() {
    a = tf.variable(tf.scalar(random(-1, 1)));
    b = tf.variable(tf.scalar(random(-1, 1)));
    c = tf.variable(tf.scalar(random(-1, 1)));
    d = tf.variable(tf.scalar(random(-1, 1)));
}

function loss(pred, labels) {
    return pred.sub(labels).square().mean();
}

function predict(x) {
    return tf.tidy(() => {
        const xs = tf.tensor1d(x);
        const ys = xs.pow(tf.scalar(3)).mul(a)
            .add(xs.square().mul(b))
            .add(xs.mul(c))
            .add(d);
        return ys;
    });
}

function trainModel() {
    if (x_vals.length > 0) {
        tf.tidy(() => {
            const ys = tf.tensor1d(y_vals);
            optimizer.minimize(() => loss(predict(x_vals), ys));
            ys.dispose();
        });
        console.log('Training model...');
    }
}

function stopPredictions() {
    predicting = false;
    console.log('Predictions stopped.');
}

let curveX, curveY;

function makePrediction() {
    if (!predicting) return;
    
    curveX = [];
    for (let x = -1; x <= 1; x += 0.05) {
        curveX.push(x);
    }
    tf.tidy(() => {
        const ys = predict(curveX);
        curveY = ys.dataSync();
    });
    console.log('Making prediction...');
}

function resetModel() {
    x_vals = [];
    y_vals = [];
    curveX = null;
    curveY = null;
    background('blue');
    predicting = true;
    initializeVariables();
    console.log('Resetting model...');
}

function mousePressed() {
    dragging = true;
}

function mouseReleased() {
    dragging = false;
}

function drawStar(x, y, radius1, radius2, npoints) {
    let angle = TWO_PI / npoints;
    let halfAngle = angle / 2.0;
    beginShape();
    for (let a = 0; a < TWO_PI; a += angle) {
        let sx = x + cos(a) * radius2;
        let sy = y + sin(a) * radius2;
        vertex(sx, sy);
        sx = x + cos(a + halfAngle) * radius1;
        sy = y + sin(a + halfAngle) * radius1;
        vertex(sx, sy);
    }
    endShape(CLOSE);
}

function draw() {
    if (dragging) {
        let x = map(mouseX, 0, width, -1, 1);
        let y = map(mouseY, 0, height, 1, -1);
        x_vals.push(x);
        y_vals.push(y);
    } else {
        tf.tidy(() => {
            if (x_vals.length > 0) {
                const ys = tf.tensor1d(y_vals);
                optimizer.minimize(() => loss(predict(x_vals), ys));
                ys.dispose(); // Ensure ys is disposed after use
            }
        });
    }
    background('blue');
    stroke(255);
    strokeWeight(8);
    for (let i = 0; i < x_vals.length; i++) {
        let px = map(x_vals[i], -1, 1, 0, width);
        let py = map(y_vals[i], -1, 1, height, 0);
        fill(255, 255, 224);
        noStroke();
        drawStar(px, py, 5, 10, 5);
    }
    if (curveX && curveY && predicting) {
        beginShape();
        noFill();
        stroke(255, 255, 224);
        strokeWeight(2);
        for (let i = 0; i < curveX.length; i++) {
            let x = map(curveX[i], -1, 1, 0, width);
            let y = map(curveY[i], -1, 1, height, 0);
            vertex(x, y);
        }
        endShape();
    }
    makePrediction();
}







