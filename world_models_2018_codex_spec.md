# World Models (2018) — Mini Reproduction / Codex Implementation Spec

**Tujuan:** membuat implementasi edukatif, reproducible, dan bertahap dari *World Models* (Ha & Schmidhuber, 2018) menggunakan **Python + PyTorch + Gymnasium CarRacing-v3**, dimulai di **MacBook Air M1 (CPU/MPS)**. Fokus awal adalah **data collection → VAE → MDN-RNN → latent/dream rollouts**. Controller adalah fase lanjutan yang **tidak boleh menghalangi MVP**.

> **Instruksi untuk Codex:** Anggap dokumen ini sebagai implementation specification. Buat code yang benar-benar bisa dieksekusi, bukan pseudocode. Kerjakan bertahap; verifikasi setiap milestone dengan smoke test dan unit test sebelum lanjut. Jangan mengklaim berhasil mereproduksi skor paper asli. Jika ada detail ambigu, pilih solusi paling sederhana, dokumentasikan pilihan tersebut, dan prioritaskan kebenaran data alignment serta mathematical correctness.

## 1. Konteks dan batasan

- **Paper:** [World Models (2018)](https://worldmodels.github.io/), *Recurrent World Models Facilitate Policy Evolution*, Ha & Schmidhuber.
- **Environment:** `CarRacing-v3`, **continuous actions** `[steering, gas, brake]`, rentang `[(-1,1), (0,1), (0,1)]`; original observations RGB `96×96×3`.
- **Preprocessing model:** resize ke `64×64`, RGB, float32 dalam `[0,1]`; network menerima format `NCHW`.
- **Hardware:** macOS Apple Silicon M1; gunakan device auto-selection **MPS → CPU**, dan opsi eksplisit `--device cpu|mps|cuda` untuk portable future runs. Tidak wajib CUDA.
- **Scope:** mini reproduction untuk memahami pipeline; bukan full reproduction compute/data/policy score tahun 2018.
- **Data awal:** smoke test 2 episode × maksimum 50 langkah; eksperimen awal 20 episode × maksimum 300 langkah; naikkan ke 50–100 episode setelah semuanya berfungsi. Ini **jumlah eksplorasi awal, bukan jaminan kualitas model**.
- **Kebijakan eksplorasi:** random continuous actions sebagai baseline; implementasikan opsi **temporally correlated random actions** (misalnya update steering/gas setiap beberapa langkah) agar dataset memiliki trajektori gerak yang lebih bermakna. Catat metode yang dipakai dalam metadata.
- **Implementasi minimum:** VAE + MDN-RNN sesuai mekanisme utama paper. Boleh menggunakan LSTM untuk setia pada paper (prioritas), bukan menggantinya dengan GRU kecuali diminta secara eksplisit.
- **VAE** dilatih pada frames; **MDN-RNN** dilatih sesudah VAE dibekukan, pada latent sequence hasil VAE.
- **Controller (CMA-ES)** hanya fase opsional setelah rollout evaluation berhasil.

## 2. Mental model dan arsitektur

### 2.1 Vision model: VAE

Observasi `o_t` adalah image dari CarRacing. VAE mempelajari posterior latent:

```text
o_t [3,64,64]
    -> Conv Encoder
    -> mu_t, logvar_t [32]
    -> z_t = mu_t + exp(0.5*logvar_t)*epsilon
    -> Conv Decoder
    -> reconstructed_o_t [3,64,64]
```

- `latent_dim = 32` (sesuai eksperimen CarRacing paper).
- Gunakan ConvVAE empat blok downsampling dan decoder transpose-conv yang simetris; jelaskan shapes dalam komentar/dokumentasi.
- Objective awal: `reconstruction MSE (atau L2) + beta * KL(q(z|o) || N(0,I))` dengan `beta` configurable. Cetak reconstruction loss dan KL **secara terpisah**.
- Gunakan output sigmoid jika rekonstruksi dinormalisasi ke `[0,1]`.
- **Jangan menuntut reconstruction error ≈ 0**. Simpan visual side-by-side rekonstruksi untuk analisis.
- VAE bekerja pada individual frames; tidak membutuhkan sequence temporal untuk melatih reconstruction.

### 2.2 Memory model: MDN-RNN

Dari episode yang sama, encode seluruh frame dengan **VAE yang sudah dilatih dan frozen**.

```text
(o_t, a_t, o_t+1)
    -> VAE frozen
(z_t, a_t, z_t+1)
    -> LSTM([z_t, a_t], recurrent_state)
    -> mixture-density output
    -> p(z_t+1 | z_t, a_t, h_t)
```

- `rnn_hidden_dim = 256`, `action_dim = 3`, `latent_dim = 32`, `num_mixtures = 5` sebagai default paper-inspired; sediakan opsi konfigurasi lebih kecil untuk M1.
- Output berupa **mixture of diagonal/factored Gaussians**, tidak sekadar MSE point prediction.
- Pilih konvensi mixture yang eksplisit, sesuai implementasi yang dipakai (misalnya **per-latent-dimension factored mixture**): `mixture_logits`, `means`, `log_stds` masing-masing shape `[B,T,D,K]` dengan `D=latent_dim`, `K=num_mixtures`. Hitung negative log-likelihood secara numerik stabil (`log_softmax`, `logsumexp`), sum/mean dimensinya konsisten, clamp `log_stds` untuk stabilitas, lalu dokumentasikan. Jangan diam-diam mencampur konvensi factored-mixture dengan single mixture index bersama untuk seluruh dimensi.
- Gunakan **teacher forcing**: input model pada timestep `t` adalah observed latent `z_t`, dan target adalah `z_(t+1)` dari actual next frame, bukan model-generated latent.
- Cache **`mu_t` dan `logvar_t` tiap frame** (dengan VAE `eval()` dan `no_grad()`) agar sample latent dapat diambil ulang tiap batch. Untuk baseline deterministik/debugging, sediakan opsi menggunakan `mu` saja.
- VAE **frozen** saat MDN-RNN training; optimizer MDN-RNN tidak mengubah encoder.
- Implementasikan independent sampling dari mixture untuk imagined rollout serta opsi deterministic diagnostic (misalnya mean atau mode-based summary, *bukan* disalahartikan sebagai sample asli).

### 2.3 Controller — optional, not MVP

Setelah VAE + MDN-RNN valid:

```text
concatenate(z_t, h_t) -> linear layer -> bounded action (steering, gas, brake)
```

- Hanya bila diminta untuk fase berikutnya: implementasikan linear policy dengan transformasi output ke action bounds.
- Optimize controller dengan **CMA-ES** atau evolutionary strategy, menggunakan reward dari simulator asli. Controller tidak dilatih dengan supervised action labels.
- **Jangan** otomatis menjalankan ribuan evaluasi simulator / mengklaim policy setara skor paper.
- Maintain RNN memory update dengan urutan yang konsisten: controller melihat `z_t` dan memory dari history; setelah memilih `a_t`, memory di-update memakai `(z_t, a_t)` untuk timestep selanjutnya.

## 3. Data pipeline: wajib benar

Gunakan Gymnasium API:

```python
obs, info = env.reset(seed=seed)
next_obs, reward, terminated, truncated, info = env.step(action)
```

**Kontrak per episode:**

- `observations`: shape `[T+1, 64, 64, 3]`, dtype `uint8`, memuat **initial observation** dan semua `next_obs`.
- `actions`: `[T, 3]`, float32.
- `rewards`: `[T]`, float32.
- `terminated`: `[T]`, bool.
- `truncated`: `[T]`, bool.
- Alignment tak boleh ambigu: `observations[t]` + `actions[t]` → `observations[t+1]`.
- Simpan satu file per episode (`.npz`, compressed). Simpan metadata JSON: seed, env ID, policy, resize convention, episode steps, collected at, versi dependencies bila tersedia.
- Jangan membuat transisi antar-episode; setiap episode independen dan hidden state LSTM harus di-reset ketika episode baru dimulai.
- Jika membuat window panjang `seq_len`, pastikan setiap window **tidak melewati episode boundary**. Untuk padding, gunakan masks agar loss tidak menghitung padding.
- Bagi **train/validation berdasarkan episode**, bukan acak frame, untuk mencegah leakage.
- Cukup satu preprocessing function yang dipakai secara identik saat train, eval, dan saat rollout.
- Hindari menyimpan seluruh dataset dalam RAM sekaligus bila tidak diperlukan (lazy episode loading / memory-conscious dataloader).

## 4. Tahapan implementasi / acceptance criteria

### Milestone A — Environment + collection

Implementasikan:

- `collect.py --episodes 2 --max-steps 50 --policy random --seed 42`
- `inspect_dataset.py` untuk memvalidasi shapes, ranges, `T+1` vs `T`, action bounds, dan contoh frames.
- Visualisasi montage dari satu episode.

**Done ketika:** smoke command berhasil, saved episode bisa dibaca kembali, transisi `(obs_t,action_t,obs_(t+1))` ter-align, dan output akhir benar tanpa runtime error.

### Milestone B — ConvVAE

Implementasikan:

- `train_vae.py --config configs/smoke.yaml` (1–2 epoch smoke, bukan target training final).
- `eval_vae.py --checkpoint ...` untuk validation loss + grid original vs reconstruction.
- Logging train/val reconstruction + KL + total loss, simpan best checkpoint.
- Unit tests: encoder `[B,3,64,64] → mu/logvar [B,32]`; decoder `[B,32] → [B,3,64,64]`; finite losses; backward gradients.

**Done ketika:** VAE training berjalan, loss finite, checkpoint reload benar, dan visual rekonstruksi tersimpan.

### Milestone C — Latent cache + MDN-RNN

Implementasikan:

- `encode_dataset.py --vae-checkpoint ...` untuk cache `mu/logvar` di setiap episode sesuai urutan frame.
- `train_mdn_rnn.py --config ... --vae-checkpoint ...`
- `eval_mdn_rnn.py` untuk held-out **one-step NLL**, plus diagnostic baseline (mis. stationary latent / simple Gaussian) bila feasible.
- Setiap sampled/derived `z[t]` dan target `z[t+1]` harus berasal dari **frame berbeda yang adjacent** dari trajectory sama, bukan `z[t]` yang salah shift.
- Reset hidden states pada boundary.
- Unit tests: output distribution shapes, valid finite log probs, NLL backward, sampling shape, no VAE gradient, no cross-episode windows.

**Done ketika:** model bisa menerima batch `[B,T,32]` + `[B,T,3]` dan mengeluarkan conditional distribution untuk `z_(t+1)`, dengan train/val NLL tercatat.

### Milestone D — Imagined rollouts + evaluation

Implementasikan:

- `rollout.py --vae-checkpoint ... --mdn-checkpoint ... --episode ... --horizon 20`
- **Teacher-forced one-step:** untuk setiap t, prediksi dari observed latents; log NLL.
- **Free-running rollout:** mulai dari observed latent dan historical LSTM state yang tersedia; lanjutkan memakai **sampled generated latent** sebagai input timestep berikutnya dan action sequence nyata yang tercatat untuk evaluasi terkontrol (open-loop action replay).
- Decode imagined latent frames dengan VAE decoder; simpan grid/GIF atau video dari **real vs imagined** dengan label jujur.
- Bandingkan horizons `1, 5, 10, 20` di held-out episodes. Jangan mengartikan pixel reconstruction drift sebagai satu-satunya metrik dynamics.
- Dengan input sampled, gunakan beberapa trajectories per start state untuk melihat stochastic variation; catat random seeds.

**Done ketika:** imagined rollout berjalan tanpa actual future observation sebagai input, hasil visual tersimpan, dan script membedakan teacher forcing vs free-running dengan jelas.

### Milestone E — Controller (deferred)

Jangan implementasikan sampai Milestones A–D selesai dan ada permintaan eksplisit untuk melanjutkan. Catat kebutuhan CMA-ES serta biaya evaluasi episode di `docs/controller_next_steps.md`.

## 5. Struktur repository yang diharapkan

```text
world-models-mini/
├── README.md
├── pyproject.toml                # dependencies + scripts, or requirements.txt
├── configs/
│   ├── smoke.yaml
│   └── baseline.yaml
├── src/world_models/
│   ├── envs.py                   # environment wrapper & image transforms
│   ├── dataset.py                # episode storage, splits, sequence loader
│   ├── vae.py                    # ConvVAE
│   ├── mdn_rnn.py                # LSTM + mixture distribution + NLL
│   ├── rollout.py                # free-run predictions
│   ├── utils.py                  # seeds, devices, checkpoints, logging
│   └── visualization.py
├── scripts/
│   ├── collect.py
│   ├── inspect_dataset.py
│   ├── train_vae.py
│   ├── eval_vae.py
│   ├── encode_dataset.py
│   ├── train_mdn_rnn.py
│   ├── eval_mdn_rnn.py
│   └── rollout.py
├── tests/
│   ├── test_data_alignment.py
│   ├── test_vae.py
│   ├── test_mdn_nll.py
│   └── test_rollout.py
├── docs/
│   └── controller_next_steps.md
└── outputs/                     # gitignored: datasets, checkpoints, plots
```

Prefer package imports (`python -m ...` atau scripts yang bisa dipanggil konsisten), type hints, explicit shape documentation, logging, progress bars, dan no-hardcoded absolute paths.

## 6. Defaults yang configurable (bukan klaim hyperparameter terbaik)

| Setting | Smoke | Baseline awal |
|---|---:|---:|
| Episodes | 2 | 50–100 |
| Max steps / episode | 50 | 300–500 |
| Resize | 64×64 | 64×64 |
| Latent dim | 32 | 32 |
| MDN-RNN hidden dim | 64 | 256 |
| Gaussian mixtures | 3 | 5 |
| Sequence length | 8 | 32 |
| Batch size | 4–8 | 16–32 (sesuaikan memory) |
| VAE epochs | 1–2 | Tunable berdasarkan val loss |
| MDN epochs | 1–2 | Tunable berdasarkan val NLL |
| Seed | 42 | 42 + tambahan untuk eksperimen |

**Device behavior:** MPS bila tersedia, fallback CPU bila tidak; pastikan operasi distribution math kompatibel; jika operasi MPS gagal, fallback yang aman didokumentasikan dan tidak menyembunyikan kegagalan training.

## 7. Kualitas dan testing yang wajib

1. Semua script memiliki `--help`, error yang mudah dipahami, dan konfigurasi tersimpan ke direktori output.
2. Reproducible train/val episode splits dengan seed; jangan split setelah membuat latent windows.
3. Checkpoint mencakup model, optimizer, step/epoch, config, seed, dan metrik terbaik.
4. MDN NLL menggunakan numerically stable `logsumexp`; test melawan distribusi Gaussian sederhana atau brute-force reference kecil.
5. Loss/gradient harus finite pada smoke test.
6. VAE harus frozen pada training MDN (test `requires_grad=False` dan tidak terjadi update bobot VAE).
7. Evaluation harus non-leaky: jangan melihat `o_(t+1)` atau latent target saat free-running rollout.
8. Semua eksperimen menghasilkan tabel CSV/JSON metrics dan setidaknya satu plot yang bisa dibuka.
9. Tidak menjalankan eksperimen besar, install kompleks, maupun latihan Controller tanpa instruksi user.
10. Jelaskan semua perbedaan implementasi mini-reproduction dengan paper asli dalam README.

## 8. Cara menjalankan (contoh target antarmuka CLI)

Perintah berikut adalah **interface yang harus dibuat oleh Codex**, bukan klaim bahwa skripnya sudah tersedia:

```bash
# Install project and Box2D dependencies in a clean virtual environment
python -m pip install -e '.[dev]'

# Sanity check environment, data, shapes
python scripts/collect.py --episodes 2 --max-steps 50 --seed 42 --policy random
python scripts/inspect_dataset.py
pytest -q

# Train VAE and inspect reconstructions
python scripts/train_vae.py --config configs/smoke.yaml
python scripts/eval_vae.py --checkpoint outputs/checkpoints/vae_best.pt

# Cache latent distributions and train dynamics
python scripts/encode_dataset.py --vae-checkpoint outputs/checkpoints/vae_best.pt
python scripts/train_mdn_rnn.py --config configs/smoke.yaml --vae-checkpoint outputs/checkpoints/vae_best.pt
python scripts/eval_mdn_rnn.py --checkpoint outputs/checkpoints/mdn_best.pt

# Imagined predictions (action replay on held-out episode)
python scripts/rollout.py --vae-checkpoint outputs/checkpoints/vae_best.pt --mdn-checkpoint outputs/checkpoints/mdn_best.pt --horizon 20
```

Codex boleh menyesuaikan lokasi file output, tetapi **harus memperbarui README dan contoh commands agar 100% sesuai** dengan interface aktual. Hindari hardcoded path jika output run mengandung timestamp atau run ID.

## 9. Apa yang tidak masuk scope awal

- Reproduksi full score CarRacing 900+.
- VizDoom, high-fidelity dream environment, atau training Controller CMA-ES besar.
- PlaNet/RSSM, latent overshooting, JEPA, Transformer dynamics, diffusion, TensorRT/Triton/CUDA custom kernels.
- Novel architecture atau paper claim sebelum baseline reproducible.
- Automated hyperparameter search atau training skala cloud.

## 10. Output yang diharapkan dari Codex pada respons pertama

1. Ringkas file yang akan dibuat dan keputusan implementasi utama.
2. Implementasikan **Milestone A** beserta smoke tests; eksekusi bila environment tersedia.
3. Jika sukses, lanjutkan Milestone B sampai D secara bertahap sesuai kemampuan environment.
4. Setelah tiap milestone, laporkan command yang dijalankan, hasil test **yang benar-benar teramati**, dan blocker yang muncul. Jangan mengarang metrics.
5. Sediakan `README.md` dengan diagram pipeline, equations, reproducibility instructions, dan limitations.
6. Hentikan sebelum Milestone E kecuali user meminta Controller.

## 11. Referensi primer

- Ha & Schmidhuber, *World Models* (2018): https://worldmodels.github.io/
- Gymnasium, *CarRacing-v3*: https://gymnasium.farama.org/environments/box2d/car_racing/
- Gymnasium, *Handling Time Limits*: https://gymnasium.farama.org/main/tutorials/handling_time_limits/
- Community PyTorch/MPS reimplementation for cross-check (bukan kode yang harus disalin): https://github.com/ml2s/world-models-2018

---

**Principle:** Mulai dari implementasi kecil yang benar secara matematis dan data alignment. Fokus utamanya adalah memahami hubungan **VAE representation → MDN-RNN conditional latent dynamics → imagined rollout**, sebelum mengerjakan Controller, ablations, atau novelty penelitian.
