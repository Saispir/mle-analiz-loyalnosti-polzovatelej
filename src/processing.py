"""Загрузка, предобработка и анализ данных Яндекс Афиши."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import URL, create_engine


class DataProcessor:
    """Инкапсулирует этапы исследования заказов Яндекс Афиши за 2024 год."""

    SQL_QUERY = """
    SELECT
        p.user_id,
        p.device_type_canonical,
        p.order_id,
        p.created_dt_msk AS order_dt,
        p.created_ts_msk AS order_ts,
        p.currency_code,
        p.revenue,
        p.tickets_count,
        EXTRACT(
            DAY FROM (
                p.created_dt_msk - LAG(p.created_dt_msk) OVER (
                    PARTITION BY p.user_id
                    ORDER BY p.created_ts_msk
                )
            )
        ) AS days_since_prev,
        p.event_id,
        e.event_name_code AS service_name,
        e.event_type_main,
        r.region_name,
        c.city_name
    FROM afisha.purchases AS p
    LEFT JOIN afisha.events AS e
        ON p.event_id = e.event_id
    LEFT JOIN afisha.city AS c
        ON e.city_id = c.city_id
    LEFT JOIN afisha.regions AS r
        ON c.region_id = r.region_id
    WHERE p.device_type_canonical IN ('mobile', 'desktop')
      AND p.created_dt_msk >= '2024-01-01'
      AND p.created_dt_msk < '2025-01-01'
      AND e.event_type_main != 'фильм'
    ORDER BY p.user_id, p.created_ts_msk;
    """

    DAYS_MAP = {
        0: "Понедельник", 1: "Вторник", 2: "Среда",
        3: "Четверг", 4: "Пятница", 5: "Суббота",
        6: "Воскресенье",
    }

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        data_dir: str = "data",
        revenue_quantile: float = 0.99,
        orders_quantile: float = 0.99,
    ):
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent.parent)
        self.data_dir = self.base_dir / data_dir
        self.revenue_quantile = revenue_quantile
        self.orders_quantile = orders_quantile
        load_dotenv(self.base_dir / ".env")

        self.engine = None
        self.df: Optional[pd.DataFrame] = None
        self.profiles: Optional[pd.DataFrame] = None
        self.segment_results: dict[str, pd.DataFrame] = {}
        self.hypothesis_results: dict[str, object] = {}
        self.phik_matrix: Optional[pd.DataFrame] = None
        self.cleaning_stats: dict[str, object] = {}

    def create_db_engine(self):
        """Создаёт SQLAlchemy engine из переменных окружения."""
        keys = {
            "username": os.getenv("DB_USER"),
            "password": os.getenv("DB_PWD"),
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT", "6432"),
            "database": os.getenv("DB_NAME", "data-analyst-afisha"),
        }
        missing = [key for key, value in keys.items() if not value]
        if missing:
            raise ValueError(
                "Не заданы переменные БД в .env: " + ", ".join(missing)
            )
        url = URL.create(
            "postgresql+psycopg2",
            username=keys["username"],
            password=keys["password"],
            host=keys["host"],
            port=int(keys["port"]),
            database=keys["database"],
        )
        return create_engine(url, pool_pre_ping=True)

    def load_data_from_db(self) -> pd.DataFrame:
        """Выгружает заказы из PostgreSQL по SQL-запросу ноутбука."""
        if self.engine is None:
            self.engine = self.create_db_engine()
        self.df = pd.read_sql_query(self.SQL_QUERY, self.engine)
        print(f"Из БД загружено строк: {len(self.df):,}")
        return self.df

    def load_data_from_csv(self, filename: str = "purchases.csv") -> pd.DataFrame:
        """Читает CSV-выгрузку заказов из папки data/."""
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Не найден файл с заказами: {path}")
        self.df = pd.read_csv(path)
        print(f"Из CSV загружено строк: {len(self.df):,}")
        return self.df

    def load_currency_data(
        self,
        filename: str = "final_tickets_tenge_df.csv",
        url: str = "https://code.s3.yandex.net/datasets/final_tickets_tenge_df.csv",
    ) -> pd.DataFrame:
        """Загружает ежедневный курс KZT; курс в источнике указан за 100 тенге."""
        path = self.data_dir / filename
        if path.exists():
            currency = pd.read_csv(path)
        else:
            try:
                currency = pd.read_csv(url)
            except Exception as error:
                raise FileNotFoundError(
                    f"Не найден {path}, загрузить курс из интернета тоже не удалось."
                ) from error
        required = {"data", "curs"}
        if not required.issubset(currency.columns):
            raise ValueError(
                "В файле курса ожидаются столбцы data и curs; "
                f"получены: {list(currency.columns)}"
            )
        currency["data"] = pd.to_datetime(currency["data"]).dt.normalize()
        currency["curs"] = pd.to_numeric(currency["curs"], errors="coerce")
        return currency[["data", "curs"]].drop_duplicates("data")

    def preprocess_data(
        self,
        df: Optional[pd.DataFrame] = None,
        currency_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Приводит типы, переводит KZT в RUB, чистит дубликаты и выбросы."""
        if df is not None:
            self.df = df.copy()
        if self.df is None:
            raise ValueError("Сначала загрузите данные методом load_data_from_db/csv.")

        data = self.df.copy()
        required = {
            "user_id", "order_id", "order_dt", "order_ts", "currency_code",
            "revenue", "tickets_count", "device_type_canonical",
            "event_type_main", "event_id", "service_name", "region_name",
        }
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"В выгрузке отсутствуют столбцы: {sorted(missing)}")

        data["order_dt"] = pd.to_datetime(data["order_dt"], errors="coerce")
        data["order_ts"] = pd.to_datetime(data["order_ts"], errors="coerce")
        data["revenue"] = pd.to_numeric(data["revenue"], errors="coerce")
        data["tickets_count"] = pd.to_numeric(
            data["tickets_count"], errors="coerce"
        )
        if "days_since_prev" in data:
            data["days_since_prev"] = pd.to_numeric(
                data["days_since_prev"], errors="coerce"
            )

        # Повторно страхуемся от фильтров исходной SQL-выгрузки.
        data = data[
            data["device_type_canonical"].isin(["mobile", "desktop"])
            & (data["order_dt"] >= "2024-01-01")
            & (data["order_dt"] < "2025-01-01")
            & (data["event_type_main"].astype(str).str.lower() != "фильм")
        ].copy()

        if currency_df is None:
            currency_df = self.load_currency_data()
        currency = currency_df.copy()
        currency["data"] = pd.to_datetime(currency["data"]).dt.normalize()
        data["_order_date"] = data["order_dt"].dt.normalize()
        data = data.merge(
            currency,
            left_on="_order_date",
            right_on="data",
            how="left",
            validate="many_to_one",
        )

        data["currency_code"] = data["currency_code"].astype(str).str.lower()
        is_kzt = data["currency_code"].eq("kzt")
        missing_kzt_rate = is_kzt & data["curs"].isna()
        if missing_kzt_rate.any():
            dates = data.loc[missing_kzt_rate, "_order_date"].nunique()
            raise ValueError(
                f"Для KZT не найден курс на {dates} датах. "
                "Проверьте файл final_tickets_tenge_df.csv."
            )

        data["revenue_rub"] = data["revenue"]
        data.loc[is_kzt, "revenue_rub"] = (
            data.loc[is_kzt, "revenue"] * data.loc[is_kzt, "curs"] / 100
        )
        data.drop(columns=["_order_date", "data", "curs"], inplace=True)

        before_dups = len(data)
        duplicate_cols = [
            "user_id", "order_ts", "device_type_canonical", "event_id",
            "service_name", "revenue_rub", "tickets_count",
        ]
        data = data.drop_duplicates(
            subset=duplicate_cols, keep="first"
        ).copy()
        implicit_duplicates = before_dups - len(data)

        before_revenue = len(data)
        data = data[
            data["revenue_rub"].notna() & (data["revenue_rub"] > 0)
        ].copy()
        revenue_threshold = data["revenue_rub"].quantile(
            self.revenue_quantile
        )
        data = data[data["revenue_rub"] <= revenue_threshold].copy()
        removed_revenue = before_revenue - len(data)

        data["tickets_count"] = pd.to_numeric(
            data["tickets_count"], downcast="integer"
        )
        data["revenue_rub"] = pd.to_numeric(
            data["revenue_rub"], downcast="float"
        )
        data = data.sort_values(
            ["user_id", "order_ts"]
        ).reset_index(drop=True)

        self.cleaning_stats = {
            "rows_after_filters_and_dedup": before_dups,
            "implicit_duplicates_removed": implicit_duplicates,
            "revenue_rows_removed": removed_revenue,
            "revenue_quantile_threshold": float(revenue_threshold),
            "rows_after_preprocessing": len(data),
        }
        self.df = data
        print(
            "Предобработка завершена: "
            f"{len(data):,} строк; неявных дубликатов удалено "
            f"{implicit_duplicates:,}; строк с неположительной/аномальной "
            f"выручкой удалено {removed_revenue:,}."
        )
        return self.df

    def create_user_profiles(self) -> pd.DataFrame:
        """Строит профиль пользователя по логике исследования из ноутбука."""
        if self.df is None:
            raise ValueError("Сначала выполните загрузку и предобработку.")
        data = self.df.sort_values(["user_id", "order_ts"]).copy()
        aggregations = {
            "first_order_dt": ("order_dt", "min"),
            "last_order_dt": ("order_dt", "max"),
            "first_device": ("device_type_canonical", "first"),
            "first_region": ("region_name", "first"),
            "first_event_type": ("event_type_main", "first"),
            "total_orders": ("order_id", "count"),
            "avg_revenue_rub": ("revenue_rub", "mean"),
            "avg_tickets_count": ("tickets_count", "mean"),
        }
        if "days_since_prev" in data:
            aggregations["avg_days_between_orders"] = (
                "days_since_prev", "mean"
            )
        profiles = data.groupby("user_id", as_index=False).agg(**aggregations)
        profiles["is_two"] = (profiles["total_orders"] >= 2).astype(int)

        # Исключение профилей выше 99-го перцентиля, как в ноутбуке.
        threshold = profiles["total_orders"].quantile(self.orders_quantile)
        before = len(profiles)
        profiles = profiles[profiles["total_orders"] <= threshold].copy()
        self.cleaning_stats["user_profiles_before_outlier_filter"] = before
        self.cleaning_stats["user_profiles_removed"] = before - len(profiles)
        self.cleaning_stats["orders_per_user_threshold"] = float(threshold)

        profiles["first_order_dt"] = pd.to_datetime(
            profiles["first_order_dt"]
        )
        profiles["first_order_dow_num"] = (
            profiles["first_order_dt"].dt.dayofweek
        )
        profiles["first_order_dow"] = profiles[
            "first_order_dow_num"
        ].map(self.DAYS_MAP)
        profiles["orders_segment"] = profiles["total_orders"].apply(
            self._orders_segment
        )
        self.profiles = profiles.reset_index(drop=True)
        print(f"Сформировано очищенных профилей: {len(self.profiles):,}")
        return self.profiles

    @staticmethod
    def _orders_segment(count: int) -> str:
        if count == 1:
            return "1 заказ"
        if count == 2:
            return "2 заказа"
        if 3 <= count <= 4:
            return "3-4 заказа"
        return "5 и более заказов"

    def calculate_retention(
        self, min_region_users: int = 50
    ) -> dict[str, pd.DataFrame]:
        """Считает долю повторных покупателей по типу события, устройству и региону."""
        if self.profiles is None:
            raise ValueError("Сначала создайте профили пользователей.")
        results = {}
        for column in ["first_event_type", "first_device", "first_region"]:
            grouped = self.profiles.groupby(
                column, dropna=False
            ).agg(
                users_count=("user_id", "count"),
                return_rate=("is_two", "mean"),
            ).reset_index()
            grouped["users_share"] = (
                grouped["users_count"] / len(self.profiles)
            )
            results[column] = grouped.sort_values(
                "users_count", ascending=False
            ).reset_index(drop=True)
        self.segment_results = results

        event = results["first_event_type"]
        sports_concerts = event[
            event["first_event_type"].astype(str).str.lower().isin(
                ["спорт", "концерты"]
            )
        ].copy()
        self.hypothesis_results["sports_vs_concerts"] = sports_concerts

        regions = results["first_region"]
        valid_regions = regions[
            regions["users_count"] >= min_region_users
        ].copy()
        if len(valid_regions) >= 2:
            spearman = valid_regions["users_count"].corr(
                valid_regions["return_rate"], method="spearman"
            )
        else:
            spearman = np.nan
        self.hypothesis_results["region_spearman"] = spearman
        self.hypothesis_results["valid_regions"] = valid_regions

        first_day = self.profiles.groupby(
            ["first_order_dow_num", "first_order_dow"], dropna=False
        ).agg(
            users_count=("user_id", "count"),
            return_rate=("is_two", "mean"),
        ).reset_index().sort_values("first_order_dow_num")
        self.segment_results["first_order_dow"] = first_day
        return results

    def correlation_analysis(self) -> pd.DataFrame:
        """Вычисляет матрицу Phi_k для признаков профиля."""
        if self.profiles is None:
            raise ValueError("Сначала создайте профили пользователей.")
        try:
            import phik  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "Для Phi_k установите зависимость: pip install phik"
            ) from error

        columns = [
            "first_device", "first_region", "first_event_type",
            "first_order_dow", "avg_revenue_rub", "avg_tickets_count",
            "total_orders", "orders_segment",
        ]
        interval_columns = [
            "avg_revenue_rub", "avg_tickets_count", "total_orders"
        ]
        self.phik_matrix = self.profiles[columns].phik_matrix(
            interval_cols=interval_columns
        )
        return self.phik_matrix

    def build_report(self) -> str:
        """Формирует краткий текстовый отчёт для терминала."""
        if self.profiles is None:
            raise ValueError("Нет профилей для отчёта.")
        p = self.profiles
        lines = [
            "\n" + "=" * 62,
            "ИТОГИ АНАЛИЗА ЛОЯЛЬНОСТИ ПОЛЬЗОВАТЕЛЕЙ ЯНДЕКС АФИШИ",
            "=" * 62,
            f"Заказов после предобработки: {len(self.df):,}",
            f"Пользователей в профилях: {len(p):,}",
            f"Средняя выручка на заказ в профиле: "
            f"{p['avg_revenue_rub'].mean():,.2f} руб.",
            f"Доля пользователей с 2+ заказами: {p['is_two'].mean():.2%}",
        ]
        if "sports_vs_concerts" in self.hypothesis_results:
            h1 = self.hypothesis_results["sports_vs_concerts"]
            lines.append("\nГипотеза 1: спорт vs концерты")
            if h1.empty:
                lines.append("В данных не найдены оба сравниваемых сегмента.")
            else:
                for _, row in h1.iterrows():
                    lines.append(
                        f"  {row['first_event_type']}: "
                        f"{row['return_rate']:.2%} "
                        f"({int(row['users_count']):,} пользователей)"
                    )
                values = {
                    str(row["first_event_type"]).lower(): row["return_rate"]
                    for _, row in h1.iterrows()
                }
                if {"спорт", "концерты"}.issubset(values):
                    conclusion = (
                        "не подтверждается"
                        if values["спорт"] <= values["концерты"]
                        else "получает описательную поддержку"
                    )
                    lines.append(f"  Гипотеза о более высокой доле спорта {conclusion}.")
        if "region_spearman" in self.hypothesis_results:
            value = self.hypothesis_results["region_spearman"]
            lines.append(
                "\nГипотеза 2: связь размера региона и возврата "
                f"(Spearman, регионы от 50 пользователей): {value:.3f}"
                if pd.notna(value)
                else "\nГипотеза 2: недостаточно регионов для расчёта Spearman."
            )
        if self.phik_matrix is not None:
            lines.append("\nСвязи Phi_k с orders_segment:")
            strongest = (
                self.phik_matrix["orders_segment"]
                .drop(labels=["orders_segment"], errors="ignore")
                .sort_values(ascending=False)
                .head(5)
            )
            lines.extend(
                f"  {name}: {value:.3f}"
                for name, value in strongest.items()
            )
        lines.extend([
            "\nИнтерпретация: Phi_k показывает силу связи, но не её направление.",
            "Рекомендации из исследования: тестировать коммуникации после",
            "первой покупки (7–14 дней), кросс-жанровые подборки,",
            "стимулы ко второму заказу и локальные кампании в регионах.",
            "Это гипотезы для A/B-тестов: LTV и ROI напрямую не оценивались.",
            "=" * 62,
        ])
        return "\n".join(lines)
