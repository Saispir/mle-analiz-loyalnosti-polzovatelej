"""Точка входа в приложение анализа Яндекс Афиши."""

from pathlib import Path

from src.processing import DataProcessor


def main():
    processor = DataProcessor(base_dir=Path(__file__).resolve().parent)

    # При наличии CSV используем сохранённую выгрузку.
    # Если её нет, подключаемся к БД через параметры из .env.
    csv_path = processor.data_dir / "purchases.csv"
    if csv_path.exists():
        processor.load_data_from_csv()
    else:
        processor.load_data_from_db()

    processor.load_currency_data()
    processor.preprocess_data()
    processor.create_user_profiles()
    processor.calculate_retention()
    processor.correlation_analysis()

    print(processor.build_report())


if __name__ == "__main__":
    main()
