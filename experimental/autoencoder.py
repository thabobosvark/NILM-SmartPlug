"""
Tiny autoencoder anomaly detector on the same top-K features.

  python autoencoder.py

K -> 8 -> 3 -> 8 -> K, trained on normal windows only (train split), then
quantised to int8 for TFLite Micro. It is class-agnostic: it learns what
"normal for any known appliance" looks like. Error = mean squared
reconstruction error in standardised units.

Outputs: ae_model.keras, ae_int8.tflite, ae_meta.json
Also exposes QuantAE, which runs the int8 model in Python so calibration
measures the SAME quantised model the ESP32 will run.
"""
import json

import joblib
import numpy as np
import tensorflow as tf

import edge_model as em


def build(k):
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(k,)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(3, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(k),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return m


def to_int8(model, Xs, rng):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep():
        for i in rng.choice(len(Xs), size=min(len(Xs), 500), replace=False):
            yield [Xs[i:i + 1].astype(np.float32)]

    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


class QuantAE:
    def __init__(self, path="ae_int8.tflite", meta="ae_meta.json"):
        m = json.load(open(meta))
        self.mu, self.sd = np.array(m["mu"]), np.array(m["sd"])
        self.it = tf.lite.Interpreter(model_path=path)
        self.it.allocate_tensors()
        self.i = self.it.get_input_details()[0]
        self.o = self.it.get_output_details()[0]

    def errors(self, F):
        """F: (n, K) reduced features in raw units -> (n,) reconstruction errors."""
        Xs = (np.asarray(F, float) - self.mu) / self.sd
        si, zi = self.i["quantization"]
        so, zo = self.o["quantization"]
        out = np.empty(len(Xs))
        for n, x in enumerate(Xs):
            q = np.clip(np.round(x / si + zi), -128, 127).astype(np.int8)[None, :]
            self.it.set_tensor(self.i["index"], q)
            self.it.invoke()
            r = (self.it.get_tensor(self.o["index"])[0].astype(np.float32) - zo) * so
            out[n] = float(np.mean((x - r) ** 2))
        return out


def main():
    tf.keras.utils.set_random_seed(42)
    rng = np.random.default_rng(42)
    stats = joblib.load("edge_stats.pkl")
    cols = stats["cols"]
    X, y, g = em.load_data()
    tr, va = em.split_masks(y, g)
    if va.sum() == 0:
        raise SystemExit("No validation windows: need more data.")
    F = X[:, cols]
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    Xs = (F - mu) / sd

    model = build(len(cols))
    model.fit(Xs[tr], Xs[tr], validation_data=(Xs[va], Xs[va]), epochs=400, batch_size=64, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True)])
    model.save("ae_model.keras")

    def err(m, A):
        return np.mean((m.predict(A, verbose=0) - A) ** 2, axis=1)

    ef = err(model, Xs[va])
    blob = to_int8(model, Xs[tr], rng)
    open("ae_int8.tflite", "wb").write(blob)
    json.dump({"mu": mu.tolist(), "sd": sd.tolist(), "k": len(cols), "cols": cols, "names": stats["names"]},
              open("ae_meta.json", "w"), indent=1)

    eq = QuantAE().errors(F[va])
    print(f"train windows {tr.sum()}, validation windows {va.sum()}")
    print(f"float  val error: median {np.median(ef):.4f}, 99th pct {np.quantile(ef, .99):.4f}")
    print(f"int8   val error: median {np.median(eq):.4f}, 99th pct {np.quantile(eq, .99):.4f}")
    print(f"float vs int8 correlation: {np.corrcoef(ef, eq)[0, 1]:.3f} (want > 0.95)")
    print(f"int8 model size: {len(blob)} bytes")
    print("Saved ae_model.keras, ae_int8.tflite, ae_meta.json")


if __name__ == "__main__":
    main()