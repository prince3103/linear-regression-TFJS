let x_axis = [];    // x axis of points marked by mouse click on canvas
let y_axis = [];    // y axis of points marked by mouse click on canvas
let m_ts;           // slope tensor
let c_ts;           // y-intercept tensor

const learningRate = 0.2;
const optimizer = tf.train.sgd(learningRate);   // Stochastic Gradient Descent
                                                // is the optimizer with a learning rate


function setup() {
  createCanvas(600, 600);
  m_ts = tf.variable(tf.scalar(random(1)));
  c_ts = tf.variable(tf.scalar(random(1)));
}

function loss(pred, labels) {
  return pred
    .sub(labels)
    .square()
    .mean();
}



function predict(x){
  const x_tsr = tf.tensor1d(x);
  const y_pred_tsr = x_tsr.mul(m_ts).add(c_ts);   // y=m^x+c
  return y_pred_tsr;
}

function mousePressed() {
  let x = map(mouseX, 0, width, 0, 1);    //x and y data-points are normalized
  let y = map(mouseY, 0, height, 1, 0);
  x_axis.push(x);
  y_axis.push(y);
}

function draw() {
  // code for training model
  if(x_axis.length>0){
    tf.tidy(() => {
        const y_train_tsr = tf.tensor1d(y_axis);
        optimizer.minimize(() => loss(predict(x_axis), y_train_tsr));
     });
  }
  
  background(0);
  // code to draw data points
  stroke(255);
  strokeWeight(8);
  for(let i =0; i<x_axis.length;i++){
    let px = map(x_axis[i], 0, 1, 0, width);
    let py = map(y_axis[i], 0, 1, height, 0);
    point(px, py);
  }
  // code for prediction
  const lineX = [0,1];
  const y_pred_tsr = tf.tidy(() => predict(lineX));
  let lineY = y_pred_tsr.dataSync();
  y_pred_tsr.dispose();

  // code to draw line
  let x1 = map(lineX[0], 0, 1, 0, width);
  let x2 = map(lineX[1], 0, 1, 0, width);
  let y1 = map(lineY[0], 0, 1, height, 0);
  let y2 = map(lineY[1], 0, 1, height, 0);

  strokeWeight(2);
  line(x1, y1, x2, y2); 
   
}