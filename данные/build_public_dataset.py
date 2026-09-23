"""Сборка обезличенного датасета для независимой проверки результатов статьи 2.

Запуск из каталога `Статья_2_эксперименты`:
    python3 данные_для_проверки/build_public_dataset.py

Что делает:
  1) строит панель «участник × выступление» тем же кодом, что и анализ в статье
     (`h2/build_panel.py`), — публикуемые данные гарантированно те же, на которых
     получены опубликованные оценки;
  2) заменяет прямые идентификаторы на псевдонимы;
  3) выгружает два файла: участники (ценности) и выступления (стратегии).

Что НЕ выгружается принципиально:
  — фамилии участников;
  — тексты реплик и цитаты-маркеры из разметки (содержат названия компаний,
    продуктов и имена, то есть опознают участников напрямую);
  — названия организаций, на площадках которых проходил сбор.

Ключ соответствия «псевдоним → фамилия» пишется в отдельный файл вне репозитория
(путь задаётся аргументом --key), чтобы авторы могли связать выгрузку с исходными
данными. Этот файл не публикуется.
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "h2"))

from build_panel import build, VALUES19, VALUES10, VALUES4, PR_EN  # noqa: E402

# Когорты нумеруются по времени сбора, названия площадок не раскрываются.
COHORT = {"avito": 1, "consortsium": 2, "MBA": 3}
PERIOD = {1: "2025-10", 2: "2026-02", 3: "2026-03"}
# Кейсы обозначаются буквами: содержательные названия к делу не относятся,
# важно лишь, что кейса два и они распределены по когортам.
CASE = {"Проект-S": "A", "Давид и Голиаф": "B"}
# Роли обозначаются нейтральными кодами слотов: содержательные названия
# персонажей различаются между версиями кейса и к анализу не относятся.
# Привязка слота к стратегии в версиях A и B РАЗНАЯ (slot_1 и slot_2 меняются
# местами) — это видно из пары колонок role_slot и prompted_strategy.
SLOT = {"Директор по маркетингу": "slot_1",
        "Вице-президент по оригинальному контенту": "slot_2",
        "Старший директор по работе с талантами и наградами": "slot_3"}

STRATEGIES = list(PR_EN.values())


def short(col: str) -> str:
    """'SDT (Самостоятельность — мысли)' -> 'SDT'; 'Self-Direction' -> 'Self-Direction'."""
    return col.split("(")[0].strip() if "(" in col else col.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE, help="куда положить csv")
    ap.add_argument("--key", default=os.path.join(HERE, "_ключ_НЕ_ПУБЛИКОВАТЬ.csv"))
    args = ap.parse_args()

    g = build()

    # ---- псевдонимы -------------------------------------------------------
    # Порядок присвоения — по когорте и алфавиту исходного идентификатора,
    # то есть воспроизводим: повторный запуск даёт те же номера.
    g["cohort"] = g.corpus.map(COHORT)
    people = (g[["pid", "cohort"]].drop_duplicates()
              .sort_values(["cohort", "pid"]).reset_index(drop=True))
    people["participant_id"] = [f"P{i + 1:03d}" for i in range(len(people))]
    pid_map = dict(zip(people.pid, people.participant_id))

    duels = (g[["duel_id", "cohort"]].drop_duplicates()
             .sort_values(["cohort", "duel_id"]).reset_index(drop=True))
    duels["duel_pid"] = [f"D{i + 1:03d}" for i in range(len(duels))]
    duel_map = dict(zip(duels.duel_id, duels.duel_pid))

    g["participant_id"] = g.pid.map(pid_map)
    g["duel"] = g.duel_id.map(duel_map)

    # ---- участники --------------------------------------------------------
    per = g.drop_duplicates("participant_id").copy()
    cols = {"participant_id": per.participant_id,
            "cohort": per.cohort,
            "collection_period": per.cohort.map(PERIOD),
            "sex": per.sex.map({"Женский": "Ж", "Мужской": "М"}),
            "age": per.age,
            "has_values": per.MRAT.notna().astype(int),
            "MRAT": per.MRAT}
    participants = pd.DataFrame(cols)
    for c in VALUES19 + VALUES10 + VALUES4:
        participants[short(c)] = per[c].values
    participants = participants.sort_values("participant_id")

    # ---- выступления ------------------------------------------------------
    perf = pd.DataFrame({
        "performance_id": [f"S{i + 1:03d}" for i in range(len(g))],
        "participant_id": g.participant_id.values,
        "cohort": g.cohort.values,
        "case": g.case.map(CASE).values,
        "duel": g.duel.values,
        "role_slot": g["Роль"].map(SLOT).values,
        "player": g.player.str.replace("Player", "", regex=False).values,
        "attempt": g["Попытка"].values,
        "prompted_strategy": g.prompted.values,
        "turns": g.turns.values,
        "n_principles_used": g.n_principles_used.values,
    })
    # ВНИМАНИЕ на именование в панели: колонка `<стратегия>` — это число ХОДОВ
    # с принципом (сумма вердиктов по ходам выступления), а `n_<стратегия>` —
    # число МАРКЕРОВ (цитат). Это разные величины, и анализ в статье опирается
    # на первую: доля ходов = <стратегия> / turns.
    for s in STRATEGIES:
        perf[f"used_{s}"] = (g[s].values > 0).astype(int)
        perf[f"turns_with_{s}"] = g[s].values
        perf[f"markers_{s}"] = g["n_" + s].values

    # ---- запись -----------------------------------------------------------
    p_out = os.path.join(args.out, "participants.csv")
    s_out = os.path.join(args.out, "performances.csv")
    participants.to_csv(p_out, index=False, encoding="utf-8")
    perf.to_csv(s_out, index=False, encoding="utf-8")

    key = people[["participant_id", "pid"]].copy()
    key.to_csv(args.key, index=False, encoding="utf-8")

    print(f"участников: {len(participants)} (с ценностями: {int(participants.has_values.sum())})")
    print(f"выступлений: {len(perf)}")
    print(f"записано: {p_out}\n          {s_out}")
    print(f"ключ (не публиковать): {args.key}")


if __name__ == "__main__":
    main()
