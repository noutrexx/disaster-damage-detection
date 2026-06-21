# Degisiklik Gunlugu

Bu dosya projedeki onemli degisiklikleri ozetler.
Format [Keep a Changelog](https://keepachangelog.com/) yaklasimini gevsek takip eder.

## [Yayinlanmamis]

### Eklenenler
- Modern dark tema arayuz: hero header, renkli metrik kartlari ve orantili dagilim barlari.
- 0-100 arasi hasar siddet endeksi paneli (Kritik / Yuksek / Dusuk).
- Guven esigi ve sinif filtreleri.
- Sonuclari PNG / CSV / JSON olarak disa aktarma.
- Saf analiz mantigi icin `core.py` modulu ve `tests/` altinda 23 birim test.
- GitHub Actions CI: her push/PR de ruff lint + pytest calisir.
- `pyproject.toml` ile pytest ve ruff yapilandirmasi.
- Gelistirme bagimliliklari icin `requirements-dev.txt`.

### Degisenler
- Analiz fonksiyonlari (`count_by_class`, `average_confidence`, `severity_index`,
  `severity_label`, `predictions_to_csv`, `extract_predictions`,
  `create_demo_predictions`) Streamlit'ten ayrilip `core.py`'ye tasindi.
- README yeni arayuz, ozellik listesi ve CI rozeti ile guncellendi.

### Duzeltilenler
- `scripts/test_model.py` icinde eksik olan `load_dotenv` importu geri eklendi
  (script `NameError` ile cokuyordu).
