// tiny_transformer.js
// 一个最简单的、纯 JavaScript、单文件、无第三方库的 Transformer 训练 + 预测示例
// 运行方式：node tiny_transformer.js

"use strict";

/**********************
 * 1. 小工具函数
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

function randomMatrix(r, c, scale = 0.1) {
  const out = [];
  for (let i = 0; i < r; i++) {
    const row = [];
    for (let j = 0; j < c; j++) {
      row.push((rand() * 2 - 1) * scale);
    }
    out.push(row);
  }
  return out;
}

function randomVector(n, scale = 0.1) {
  const out = [];
  for (let i = 0; i < n; i++) out.push((rand() * 2 - 1) * scale);
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

function addVec(a, b) {
  const out = [];
  for (let i = 0; i < a.length; i++) out.push(a[i] + b[i]);
  return out;
}

function subVec(a, b) {
  const out = [];
  for (let i = 0; i < a.length; i++) out.push(a[i] - b[i]);
  return out;
}

function scaleVec(a, s) {
  const out = [];
  for (let i = 0; i < a.length; i++) out.push(a[i] * s);
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

function matVecMul(a, v) {
  const r = a.length;
  const c = a[0].length;
  const out = zeros(r);
  for (let i = 0; i < r; i++) {
    let s = 0;
    for (let j = 0; j < c; j++) s += a[i][j] * v[j];
    out[i] = s;
  }
  return out;
}

function vecMatMul(v, m) {
  const r = m.length;
  const c = m[0].length;
  const out = zeros(c);
  for (let j = 0; j < c; j++) {
    let s = 0;
    for (let i = 0; i < r; i++) s += v[i] * m[i][j];
    out[j] = s;
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

function outer(a, b) {
  const out = zeros2D(a.length, b.length);
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) out[i][j] = a[i] * b[j];
  }
  return out;
}

function addInPlace2D(a, b) {
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[0].length; j++) a[i][j] += b[i][j];
  }
}

function addInPlace1D(a, b) {
  for (let i = 0; i < a.length; i++) a[i] += b[i];
}

function scaleInPlace2D(a, s) {
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[0].length; j++) a[i][j] *= s;
  }
}

function scaleInPlace1D(a, s) {
  for (let i = 0; i < a.length; i++) a[i] *= s;
}

function argmax(v) {
  let idx = 0;
  for (let i = 1; i < v.length; i++) {
    if (v[i] > v[idx]) idx = i;
  }
  return idx;
}

function crossEntropyLoss(probs, target) {
  const eps = 1e-12;
  return -Math.log(probs[target] + eps);
}

/**********************
 * 2. 数据：全部写在一个文件里
 **********************/
// 任务：输入 3 个 0/1，预测其和的奇偶性
// label = (x1 + x2 + x3) % 2

const dataset = [
  { x: [0, 0, 0], y: 0 },
  { x: [0, 0, 1], y: 1 },
  { x: [0, 1, 0], y: 1 },
  { x: [0, 1, 1], y: 0 },
  { x: [1, 0, 0], y: 1 },
  { x: [1, 0, 1], y: 0 },
  { x: [1, 1, 0], y: 0 },
  { x: [1, 1, 1], y: 1 }
];

/**********************
 * 3. 模型参数
 **********************/
const vocabSize = 2;   // token 只有 0 和 1
const seqLen = 3;      // 每个样本长度是 3
const dModel = 8;      // embedding 维度
const dHidden = 16;    // FFN 隐层
const numClasses = 2;  // 输出 0 或 1

const model = {
  tokenEmb: randomMatrix(vocabSize, dModel, 0.2),
  posEmb: randomMatrix(seqLen, dModel, 0.2),

  Wq: randomMatrix(dModel, dModel, 0.2),
  Wk: randomMatrix(dModel, dModel, 0.2),
  Wv: randomMatrix(dModel, dModel, 0.2),

  W1: randomMatrix(dModel, dHidden, 0.2),
  b1: randomVector(dHidden, 0.0),

  W2: randomMatrix(dHidden, dModel, 0.2),
  b2: randomVector(dModel, 0.0),

  Wo: randomMatrix(dModel, numClasses, 0.2),
  bo: randomVector(numClasses, 0.0)
};

/**********************
 * 4. 前向传播
 **********************/
function forward(sample, model) {
  const xTokens = sample.x; // [0,1,0] 这种

  // (1) embedding + position
  const X = zeros2D(seqLen, dModel);
  for (let t = 0; t < seqLen; t++) {
    const tokenId = xTokens[t];
    for (let j = 0; j < dModel; j++) {
      X[t][j] = model.tokenEmb[tokenId][j] + model.posEmb[t][j];
    }
  }

  // (2) self-attention
  const Q = matMul(X, model.Wq); // [seq, d]
  const K = matMul(X, model.Wk); // [seq, d]
  const V = matMul(X, model.Wv); // [seq, d]

  const scale = 1 / Math.sqrt(dModel);

  const scores = zeros2D(seqLen, seqLen);
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      scores[i][j] = dot(Q[i], K[j]) * scale;
    }
  }

  const A = softmaxRows(scores); // attention 权重 [seq, seq]
  const H = matMul(A, V);        // [seq, d]

  // residual
  const X2 = addMat(X, H);

  // (3) FFN
  const Z1 = addBiasRows(matMul(X2, model.W1), model.b1); // [seq, hidden]
  const A1 = relu2D(Z1);
  const Z2 = addBiasRows(matMul(A1, model.W2), model.b2); // [seq, d]

  // residual
  const Y = addMat(X2, Z2);

  // (4) mean pool + classifier
  const pooled = meanRows(Y); // [d]
  const logits = vecMatMul(pooled, model.Wo); // [2]
  for (let i = 0; i < logits.length; i++) logits[i] += model.bo[i];

  const probs = softmax(logits);
  const loss = crossEntropyLoss(probs, sample.y);

  return {
    loss,
    probs,
    pred: argmax(probs),
    cache: {
      xTokens,
      X, Q, K, V, scores, A, H, X2, Z1, A1, Z2, Y, pooled, logits
    }
  };
}

/**********************
 * 5. 反向传播
 **********************/
function backward(sample, model, cache) {
  const grads = {
    tokenEmb: zeros2D(vocabSize, dModel),
    posEmb: zeros2D(seqLen, dModel),

    Wq: zeros2D(dModel, dModel),
    Wk: zeros2D(dModel, dModel),
    Wv: zeros2D(dModel, dModel),

    W1: zeros2D(dModel, dHidden),
    b1: zeros(dHidden),

    W2: zeros2D(dHidden, dModel),
    b2: zeros(dModel),

    Wo: zeros2D(dModel, numClasses),
    bo: zeros(numClasses)
  };

  const {
    xTokens,
    X, Q, K, V, A, H, X2, Z1, A1, Z2, Y, pooled, probs
  } = {
    ...cache,
    probs: softmax(cache.logits)
  };

  // dLoss / dLogits = probs - onehot(target)
  const dLogits = probs.slice();
  dLogits[sample.y] -= 1; // [2]

  // classifier
  const dWo = outer(pooled, dLogits); // [d,2]
  addInPlace2D(grads.Wo, dWo);
  addInPlace1D(grads.bo, dLogits);

  // dPooled = Wo * dLogits
  const dPooled = zeros(dModel);
  for (let i = 0; i < dModel; i++) {
    let s = 0;
    for (let j = 0; j < numClasses; j++) s += model.Wo[i][j] * dLogits[j];
    dPooled[i] = s;
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
  // dW2 = A1^T * dZ2
  addInPlace2D(grads.W2, matMul(transpose(A1), dZ2));

  // db2 = sum rows
  for (let j = 0; j < dModel; j++) {
    let s = 0;
    for (let t = 0; t < seqLen; t++) s += dZ2[t][j];
    grads.b2[j] += s;
  }

  // dA1 = dZ2 * W2^T
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
  // dA = dH * V^T
  const dA = matMul(dH, transpose(V));

  // dV = A^T * dH
  const dV = matMul(transpose(A), dH);

  // A = softmaxRows(scores)
  const dScores = zeros2D(seqLen, seqLen);
  for (let i = 0; i < seqLen; i++) {
    // row-wise softmax backward:
    // ds = a * (da - sum(da * a))
    let rowDot = 0;
    for (let j = 0; j < seqLen; j++) rowDot += dA[i][j] * A[i][j];
    for (let j = 0; j < seqLen; j++) {
      dScores[i][j] = A[i][j] * (dA[i][j] - rowDot);
    }
  }

  const scale = 1 / Math.sqrt(dModel);

  // scores[i][j] = dot(Q[i], K[j]) * scale
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
  const dX_from_q = matMul(dQ, transpose(model.Wq));
  addInPlace2D(dX, dX_from_q);

  // K = X * Wk
  addInPlace2D(grads.Wk, matMul(transpose(X), dK));
  const dX_from_k = matMul(dK, transpose(model.Wk));
  addInPlace2D(dX, dX_from_k);

  // V = X * Wv
  addInPlace2D(grads.Wv, matMul(transpose(X), dV));
  const dX_from_v = matMul(dV, transpose(model.Wv));
  addInPlace2D(dX, dX_from_v);

  // X = tokenEmb[token] + posEmb[pos]
  for (let t = 0; t < seqLen; t++) {
    const tokenId = xTokens[t];
    for (let j = 0; j < dModel; j++) {
      grads.tokenEmb[tokenId][j] += dX[t][j];
      grads.posEmb[t][j] += dX[t][j];
    }
  }

  return grads;
}

/**********************
 * 6. 参数更新
 **********************/
function sgdStep(model, grads, lr) {
  // tokenEmb
  for (let i = 0; i < vocabSize; i++) {
    for (let j = 0; j < dModel; j++) {
      model.tokenEmb[i][j] -= lr * grads.tokenEmb[i][j];
    }
  }

  // posEmb
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < dModel; j++) {
      model.posEmb[i][j] -= lr * grads.posEmb[i][j];
    }
  }

  // Wq Wk Wv
  const mats = ["Wq", "Wk", "Wv"];
  for (const name of mats) {
    for (let i = 0; i < dModel; i++) {
      for (let j = 0; j < dModel; j++) {
        model[name][i][j] -= lr * grads[name][i][j];
      }
    }
  }

  // W1 b1
  for (let i = 0; i < dModel; i++) {
    for (let j = 0; j < dHidden; j++) {
      model.W1[i][j] -= lr * grads.W1[i][j];
    }
  }
  for (let j = 0; j < dHidden; j++) model.b1[j] -= lr * grads.b1[j];

  // W2 b2
  for (let i = 0; i < dHidden; i++) {
    for (let j = 0; j < dModel; j++) {
      model.W2[i][j] -= lr * grads.W2[i][j];
    }
  }
  for (let j = 0; j < dModel; j++) model.b2[j] -= lr * grads.b2[j];

  // Wo bo
  for (let i = 0; i < dModel; i++) {
    for (let j = 0; j < numClasses; j++) {
      model.Wo[i][j] -= lr * grads.Wo[i][j];
    }
  }
  for (let j = 0; j < numClasses; j++) model.bo[j] -= lr * grads.bo[j];
}

/**********************
 * 7. 训练
 **********************/
function shuffleInPlace(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
}

function train(model, data, epochs = 2000, lr = 0.03) {
  for (let epoch = 1; epoch <= epochs; epoch++) {
    shuffleInPlace(data);

    let totalLoss = 0;
    let correct = 0;

    for (const sample of data) {
      const out = forward(sample, model);
      totalLoss += out.loss;
      if (out.pred === sample.y) correct++;

      const grads = backward(sample, model, out.cache);
      sgdStep(model, grads, lr);
    }

    if (epoch % 200 === 0 || epoch === 1) {
      const avgLoss = totalLoss / data.length;
      const acc = correct / data.length;
      console.log(
        `epoch=${epoch} loss=${avgLoss.toFixed(4)} acc=${(acc * 100).toFixed(1)}%`
      );
    }
  }
}

/**********************
 * 8. 预测
 **********************/
function predict(model, x) {
  const sample = { x, y: 0 }; // y 在预测时无实际作用
  const out = forward(sample, model);
  return {
    input: x,
    probs: out.probs,
    pred: out.pred
  };
}

/**********************
 * 9. 主程序
 **********************/
console.log("开始训练...");
train(model, dataset, 2000, 0.03);

console.log("\n训练后对全部样本进行预测：");
for (const sample of dataset) {
  const result = predict(model, sample.x);
  console.log(
    `input=${JSON.stringify(sample.x)} ` +
    `label=${sample.y} ` +
    `pred=${result.pred} ` +
    `probs=${result.probs.map(v => v.toFixed(4)).join(", ")}`
  );
}

console.log("\n单独测试几个输入：");
const tests = [
  [0, 0, 1],
  [1, 1, 0],
  [1, 1, 1],
  [0, 0, 0]
];

for (const x of tests) {
  const result = predict(model, x);
  console.log(
    `predict(${JSON.stringify(x)}) => ${result.pred}, probs=${result.probs.map(v => v.toFixed(4)).join(", ")}`
  );
}
