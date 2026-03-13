// tiny_transformer_regression.js
// 纯 JS、单文件、无第三方库
// 一个最简单的 Transformer 回归例子：输入 3 个数，输出一个实数

"use strict";

/**********************
 * 1. 工具函数
 **********************/
function createSeededRandom(seed) {
  let s = seed >>> 0;
  return function rand() {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

const rand = createSeededRandom(42);

function zeros(n) {
  return Array(n).fill(0);
}

function zeros2D(r, c) {
  const out = [];
  for (let i = 0; i < r; i++) out.push(zeros(c));
  return out;
}

function randomVector(n, scale = 0.1) {
  const out = [];
  for (let i = 0; i < n; i++) out.push((rand() * 2 - 1) * scale);
  return out;
}

function randomMatrix(r, c, scale = 0.1) {
  const out = [];
  for (let i = 0; i < r; i++) {
    const row = [];
    for (let j = 0; j < c; j++) row.push((rand() * 2 - 1) * scale);
    out.push(row);
  }
  return out;
}

function clone2D(a) {
  return a.map(row => row.slice());
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function transpose(a) {
  const r = a.length;
  const c = a[0].length;
  const out = zeros2D(c, r);
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) {
      out[j][i] = a[i][j];
    }
  }
  return out;
}

function matMul(a, b) {
  const r = a.length;
  const k = a[0].length;
  const c = b[0].length;
  const out = zeros2D(r, c);
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) {
      let s = 0;
      for (let t = 0; t < k; t++) s += a[i][t] * b[t][j];
      out[i][j] = s;
    }
  }
  return out;
}

function addMat(a, b) {
  const out = [];
  for (let i = 0; i < a.length; i++) {
    const row = [];
    for (let j = 0; j < a[0].length; j++) row.push(a[i][j] + b[i][j]);
    out.push(row);
  }
  return out;
}

function addBiasRows(x, b) {
  const out = [];
  for (let i = 0; i < x.length; i++) {
    const row = [];
    for (let j = 0; j < x[0].length; j++) row.push(x[i][j] + b[j]);
    out.push(row);
  }
  return out;
}

function relu2D(x) {
  const out = [];
  for (let i = 0; i < x.length; i++) {
    const row = [];
    for (let j = 0; j < x[0].length; j++) row.push(x[i][j] > 0 ? x[i][j] : 0);
    out.push(row);
  }
  return out;
}

function softmax(vec) {
  let maxV = -Infinity;
  for (let i = 0; i < vec.length; i++) if (vec[i] > maxV) maxV = vec[i];
  const exps = [];
  let sum = 0;
  for (let i = 0; i < vec.length; i++) {
    const e = Math.exp(vec[i] - maxV);
    exps.push(e);
    sum += e;
  }
  return exps.map(v => v / sum);
}

function softmaxRows(x) {
  return x.map(row => softmax(row));
}

function meanRows(x) {
  const r = x.length;
  const c = x[0].length;
  const out = zeros(c);
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) out[j] += x[i][j];
  }
  for (let j = 0; j < c; j++) out[j] /= r;
  return out;
}

function addInPlace1D(a, b) {
  for (let i = 0; i < a.length; i++) a[i] += b[i];
}

function addInPlace2D(a, b) {
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[0].length; j++) a[i][j] += b[i][j];
  }
}

function shuffleInPlace(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
}

/**********************
 * 2. 数据（全部写在这里）
 **********************/
// 任务：给 3 个输入数，回归预测
// y = 0.2*x0 + 0.3*x1 + 0.5*x2
//
// 这里数据量故意很小
const dataset = [
  { x: [0.0, 0.0, 0.0], y: 0.0 },
  { x: [1.0, 0.0, 0.0], y: 0.2 },
  { x: [0.0, 1.0, 0.0], y: 0.3 },
  { x: [0.0, 0.0, 1.0], y: 0.5 },
  { x: [1.0, 1.0, 0.0], y: 0.5 },
  { x: [1.0, 0.0, 1.0], y: 0.7 },
  { x: [0.0, 1.0, 1.0], y: 0.8 },
  { x: [1.0, 1.0, 1.0], y: 1.0 },
  { x: [0.2, 0.4, 0.6], y: 0.46 },
  { x: [0.9, 0.1, 0.3], y: 0.36 },
  { x: [0.5, 0.2, 0.8], y: 0.56 },
  { x: [0.3, 0.7, 0.9], y: 0.72 }
];

/**********************
 * 3. 模型参数
 **********************/
const seqLen = 3;
const dModel = 8;
const dHidden = 16;

const model = {
  // 把每个标量 x_t 映射到 dModel 维向量
  valueW: randomVector(dModel, 0.2), // x_t * valueW[j]
  valueB: randomVector(dModel, 0.2), // + valueB[j]

  // 位置编码
  posEmb: randomMatrix(seqLen, dModel, 0.2),

  // attention
  Wq: randomMatrix(dModel, dModel, 0.2),
  Wk: randomMatrix(dModel, dModel, 0.2),
  Wv: randomMatrix(dModel, dModel, 0.2),

  // FFN
  W1: randomMatrix(dModel, dHidden, 0.2),
  b1: randomVector(dHidden, 0.0),
  W2: randomMatrix(dHidden, dModel, 0.2),
  b2: randomVector(dModel, 0.0),

  // 回归头：输出 1 个数字
  Wo: randomVector(dModel, 0.2),
  bo: 0
};

/**********************
 * 4. 前向传播
 **********************/
function forward(sample, model) {
  const x = sample.x; // 长度为 3 的数值序列

  // (1) 数值输入 -> embedding
  const X = zeros2D(seqLen, dModel);
  for (let t = 0; t < seqLen; t++) {
    for (let j = 0; j < dModel; j++) {
      X[t][j] = x[t] * model.valueW[j] + model.valueB[j] + model.posEmb[t][j];
    }
  }

  // (2) Self-Attention
  const Q = matMul(X, model.Wq);
  const K = matMul(X, model.Wk);
  const V = matMul(X, model.Wv);

  const scale = 1 / Math.sqrt(dModel);
  const scores = zeros2D(seqLen, seqLen);

  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      scores[i][j] = dot(Q[i], K[j]) * scale;
    }
  }

  const A = softmaxRows(scores);
  const H = matMul(A, V);

  // residual
  const X2 = addMat(X, H);

  // (3) FFN
  const Z1 = addBiasRows(matMul(X2, model.W1), model.b1);
  const A1 = relu2D(Z1);
  const Z2 = addBiasRows(matMul(A1, model.W2), model.b2);

  // residual
  const Y = addMat(X2, Z2);

  // (4) mean pooling
  const pooled = meanRows(Y);

  // (5) 输出一个实数
  let pred = 0;
  for (let j = 0; j < dModel; j++) pred += pooled[j] * model.Wo[j];
  pred += model.bo;

  // 回归损失：MSE 的 1/2 形式
  const diff = pred - sample.y;
  const loss = 0.5 * diff * diff;

  return {
    pred,
    loss,
    cache: {
      x,
      X, Q, K, V, scores, A, H, X2, Z1, A1, Z2, Y, pooled, pred
    }
  };
}

/**********************
 * 5. 反向传播
 **********************/
function backward(sample, model, cache) {
  const grads = {
    valueW: zeros(dModel),
    valueB: zeros(dModel),
    posEmb: zeros2D(seqLen, dModel),

    Wq: zeros2D(dModel, dModel),
    Wk: zeros2D(dModel, dModel),
    Wv: zeros2D(dModel, dModel),

    W1: zeros2D(dModel, dHidden),
    b1: zeros(dHidden),
    W2: zeros2D(dHidden, dModel),
    b2: zeros(dModel),

    Wo: zeros(dModel),
    bo: 0
  };

  const { x, X, Q, K, V, A, X2, Z1, A1, Y, pooled, pred } = cache;

  // loss = 0.5 * (pred - y)^2
  // dLoss/dPred = pred - y
  const dPred = pred - sample.y;

  // pred = dot(pooled, Wo) + bo
  for (let j = 0; j < dModel; j++) {
    grads.Wo[j] += pooled[j] * dPred;
  }
  grads.bo += dPred;

  const dPooled = zeros(dModel);
  for (let j = 0; j < dModel; j++) {
    dPooled[j] = model.Wo[j] * dPred;
  }

  // pooled = meanRows(Y)
  const dY = zeros2D(seqLen, dModel);
  for (let t = 0; t < seqLen; t++) {
    for (let j = 0; j < dModel; j++) {
      dY[t][j] = dPooled[j] / seqLen;
    }
  }

  // Y = X2 + Z2
  const dX2 = clone2D(dY);
  const dZ2 = clone2D(dY);

  // Z2 = A1 * W2 + b2
  addInPlace2D(grads.W2, matMul(transpose(A1), dZ2));

  for (let j = 0; j < dModel; j++) {
    let s = 0;
    for (let t = 0; t < seqLen; t++) s += dZ2[t][j];
    grads.b2[j] += s;
  }

  const dA1 = matMul(dZ2, transpose(model.W2));

  // A1 = relu(Z1)
  const dZ1 = zeros2D(seqLen, dHidden);
  for (let t = 0; t < seqLen; t++) {
    for (let j = 0; j < dHidden; j++) {
      dZ1[t][j] = Z1[t][j] > 0 ? dA1[t][j] : 0;
    }
  }

  // Z1 = X2 * W1 + b1
  addInPlace2D(grads.W1, matMul(transpose(X2), dZ1));

  for (let j = 0; j < dHidden; j++) {
    let s = 0;
    for (let t = 0; t < seqLen; t++) s += dZ1[t][j];
    grads.b1[j] += s;
  }

  const dX2_from_ffn = matMul(dZ1, transpose(model.W1));
  addInPlace2D(dX2, dX2_from_ffn);

  // X2 = X + H
  const dX = clone2D(dX2);
  const dH = clone2D(dX2);

  // H = A * V
  const dA = matMul(dH, transpose(V));
  const dV = matMul(transpose(A), dH);

  // A = softmaxRows(scores)
  const dScores = zeros2D(seqLen, seqLen);
  for (let i = 0; i < seqLen; i++) {
    let rowDot = 0;
    for (let j = 0; j < seqLen; j++) rowDot += dA[i][j] * A[i][j];
    for (let j = 0; j < seqLen; j++) {
      dScores[i][j] = A[i][j] * (dA[i][j] - rowDot);
    }
  }

  const scale = 1 / Math.sqrt(dModel);
  const dQ = zeros2D(seqLen, dModel);
  const dK = zeros2D(seqLen, dModel);

  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      const ds = dScores[i][j] * scale;
      for (let k = 0; k < dModel; k++) {
        dQ[i][k] += ds * K[j][k];
        dK[j][k] += ds * Q[i][k];
      }
    }
  }

  // Q = X * Wq
  addInPlace2D(grads.Wq, matMul(transpose(X), dQ));
  addInPlace2D(dX, matMul(dQ, transpose(model.Wq)));

  // K = X * Wk
  addInPlace2D(grads.Wk, matMul(transpose(X), dK));
  addInPlace2D(dX, matMul(dK, transpose(model.Wk)));

  // V = X * Wv
  addInPlace2D(grads.Wv, matMul(transpose(X), dV));
  addInPlace2D(dX, matMul(dV, transpose(model.Wv)));

  // X[t][j] = x[t] * valueW[j] + valueB[j] + posEmb[t][j]
  for (let t = 0; t < seqLen; t++) {
    for (let j = 0; j < dModel; j++) {
      grads.valueW[j] += x[t] * dX[t][j];
      grads.valueB[j] += dX[t][j];
      grads.posEmb[t][j] += dX[t][j];
    }
  }

  return grads;
}

/**********************
 * 6. 参数更新
 **********************/
function sgdStep(model, grads, lr) {
  for (let j = 0; j < dModel; j++) {
    model.valueW[j] -= lr * grads.valueW[j];
    model.valueB[j] -= lr * grads.valueB[j];
  }

  for (let t = 0; t < seqLen; t++) {
    for (let j = 0; j < dModel; j++) {
      model.posEmb[t][j] -= lr * grads.posEmb[t][j];
    }
  }

  for (let i = 0; i < dModel; i++) {
    for (let j = 0; j < dModel; j++) {
      model.Wq[i][j] -= lr * grads.Wq[i][j];
      model.Wk[i][j] -= lr * grads.Wk[i][j];
      model.Wv[i][j] -= lr * grads.Wv[i][j];
    }
  }

  for (let i = 0; i < dModel; i++) {
    for (let j = 0; j < dHidden; j++) {
      model.W1[i][j] -= lr * grads.W1[i][j];
    }
  }
  for (let j = 0; j < dHidden; j++) {
    model.b1[j] -= lr * grads.b1[j];
  }

  for (let i = 0; i < dHidden; i++) {
    for (let j = 0; j < dModel; j++) {
      model.W2[i][j] -= lr * grads.W2[i][j];
    }
  }
  for (let j = 0; j < dModel; j++) {
    model.b2[j] -= lr * grads.b2[j];
  }

  for (let j = 0; j < dModel; j++) {
    model.Wo[j] -= lr * grads.Wo[j];
  }
  model.bo -= lr * grads.bo;
}

/**********************
 * 7. 训练
 **********************/
function train(model, data, epochs = 2000, lr = 0.01) {
  for (let epoch = 1; epoch <= epochs; epoch++) {
    shuffleInPlace(data);

    let totalLoss = 0;

    for (const sample of data) {
      const out = forward(sample, model);
      totalLoss += out.loss;

      const grads = backward(sample, model, out.cache);
      sgdStep(model, grads, lr);
    }

    if (epoch % 200 === 0 || epoch === 1) {
      console.log(
        `epoch=${epoch} avg_loss=${(totalLoss / data.length).toFixed(6)}`
      );
    }
  }
}

/**********************
 * 8. 预测
 **********************/
function predict(model, x) {
  const out = forward({ x, y: 0 }, model);
  return out.pred;
}

/**********************
 * 9. 主程序
 **********************/
console.log("开始训练...");
train(model, dataset, 2000, 0.01);

console.log("\n训练集预测结果：");
for (const sample of dataset) {
  const pred = predict(model, sample.x);
  console.log(
    `x=${JSON.stringify(sample.x)} ` +
    `target=${sample.y.toFixed(4)} ` +
    `pred=${pred.toFixed(4)}`
  );
}

console.log("\n测试几个新样本：");
const tests = [
  [0.1, 0.2, 0.3], // 真值 0.23
  [0.7, 0.4, 0.9], // 真值 0.71
  [0.9, 0.9, 0.1], // 真值 0.50
  [0.2, 0.8, 0.5]  // 真值 0.53
];

for (const x of tests) {
  const pred = predict(model, x);
  const target = 0.2 * x[0] + 0.3 * x[1] + 0.5 * x[2];
  console.log(
    `x=${JSON.stringify(x)} target=${target.toFixed(4)} pred=${pred.toFixed(4)}`
  );
}
