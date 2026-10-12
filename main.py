import os
import time
import csv
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ========== КОНФИГУРАЦИЯ ==========
API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
MODEL = os.getenv("LLM_MODEL")
RESULTS_FILE = "results.csv"


# ========== БАЗОВЫЙ КЛАСС ==========
class BaseExperiment:
    """Общая логика: клиент OpenAI, вызов модели, замер времени, обработка ошибок."""

    name = "base"
    description = "Базовый эксперимент"

    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.results = []

    def ask_model(self, prompt: str, system: str,
                  temperature=None, max_tokens=None):
        kwargs = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(**kwargs)
            latency = time.perf_counter() - started
            return response.choices[0].message.content, latency, None
        except Exception as e:
            latency = time.perf_counter() - started
            return None, latency, str(e)

    def run(self):
        raise NotImplementedError("Метод run() должен быть переопределён")

    def get_results(self):
        return self.results


# ========== ЭКСПЕРИМЕНТ 1: SYSTEM PROMPT ==========
class SystemPromptExperiment(BaseExperiment):
    name = "system_prompt"
    description = "Влияние System Prompt на структуру и роль ответа"

    USER_PROMPT = (
        "Событие: в 03:17 зафиксировано 25 неудачных попыток входа "
        "в учётную запись admin с одного IP, после чего произошёл "
        "успешный вход. Проанализируй ситуацию."
    )

    SYSTEMS = [
        {"id": "sys_01",
         "text": "Отвечай на запрос пользователя."},
        {"id": "sys_02",
         "text": ("Ты ассистент аналитика ИБ. Отвечай не более чем в 5 пунктах. "
                  "Для каждого пункта укажи риск и рекомендуемое действие.")},
        {"id": "sys_03",
         "text": ("Ты ассистент аналитика ИБ. Не выдумывай недостающие данные. "
                  "Явно отделяй факты от предположений. Ответ верни в формате: "
                  "Факты / Гипотезы / Действия.")},
    ]

    def run(self):
        print(f"\n{'=' * 60}")
        print(f"ЭКСПЕРИМЕНТ 1: {self.description}")
        print('=' * 60)

        for sys_item in self.SYSTEMS:
            print(f"\n[{sys_item['id']}] {sys_item['text'][:70]}...")
            answer, latency, error = self.ask_model(
                prompt=self.USER_PROMPT,
                system=sys_item["text"],
                temperature=0.7,
            )
            if error:
                print(f"  ОШИБКА: {error}")
            else:
                print(f"  OK ({latency:.2f} с), длина: {len(answer)} симв.")

            self.results.append({
                "experiment": self.name,
                "model": MODEL,
                "system_id": sys_item["id"],
                "system_text": sys_item["text"],
                "user_prompt": self.USER_PROMPT,
                "temperature": 0.7,
                "max_tokens": None,
                "repeat": 1,
                "latency": round(latency, 3),
                "answer": answer,
                "error": error,
            })


# ========== ЭКСПЕРИМЕНТ 2: TEMPERATURE ==========
class TemperatureExperiment(BaseExperiment):
    name = "temperature"
    description = "Влияние Temperature на вариативность ответов"

    USER_PROMPT = (
        "Предложи пять названий сервиса для автоматического анализа "
        "журналов событий информационной безопасности. Для каждого "
        "названия дай пояснение в одном предложении."
    )
    SYSTEM_PROMPT = "Ты креативный помощник."
    TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
    REPEATS = 5

    def run(self):
        print(f"\n{'=' * 60}")
        print(f"ЭКСПЕРИМЕНТ 2: {self.description}")
        print(f"Всего запросов: {len(self.TEMPERATURES) * self.REPEATS}")
        print('=' * 60)

        for temp in self.TEMPERATURES:
            print(f"\n--- Temperature = {temp} ---")
            for repeat in range(1, self.REPEATS + 1):
                answer, latency, error = self.ask_model(
                    prompt=self.USER_PROMPT,
                    system=self.SYSTEM_PROMPT,
                    temperature=temp,
                )
                if error:
                    print(f"  [{repeat}/{self.REPEATS}] ОШИБКА: {error}")
                else:
                    print(f"  [{repeat}/{self.REPEATS}] OK ({latency:.2f} с)")

                self.results.append({
                    "experiment": self.name,
                    "model": MODEL,
                    "system_id": "sys_creative",
                    "system_text": self.SYSTEM_PROMPT,
                    "user_prompt": self.USER_PROMPT,
                    "temperature": temp,
                    "max_tokens": None,
                    "repeat": repeat,
                    "latency": round(latency, 3),
                    "answer": answer,
                    "error": error,
                })


# ========== ЭКСПЕРИМЕНТ 3: MAX TOKENS ==========
class MaxTokensExperiment(BaseExperiment):
    name = "max_tokens"
    description = "Влияние ограничения длины ответа"

    USER_PROMPT = (
        "Объясни студенту 4 курса архитектуру RAG-системы: ingestion, "
        "chunking, embeddings, vector database, retrieval, prompt "
        "construction и generation. Для каждого этапа укажи его "
        "назначение и одну типичную ошибку."
    )
    SYSTEM_PROMPT = ("Ты технический преподаватель. "
                     "Отвечай подробно и структурированно.")
    MAX_TOKENS_VALUES = [100, 300, 1000]

    def run(self):
        print(f"\n{'=' * 60}")
        print(f"ЭКСПЕРИМЕНТ 3: {self.description}")
        print('=' * 60)

        for max_tok in self.MAX_TOKENS_VALUES:
            print(f"\n--- max_tokens = {max_tok} ---")
            answer, latency, error = self.ask_model(
                prompt=self.USER_PROMPT,
                system=self.SYSTEM_PROMPT,
                temperature=0.7,
                max_tokens=max_tok,
            )
            if error:
                print(f"  ОШИБКА: {error}")
            else:
                print(f"  OK ({latency:.2f} с), длина ответа: {len(answer)} символов")
                # Проверка обрыва
                if not answer.rstrip().endswith(('.', '!', '?', '`', ':')):
                    print(f"  ⚠ Возможен обрыв ответа.")

            self.results.append({
                "experiment": self.name,
                "model": MODEL,
                "system_id": "sys_teacher",
                "system_text": self.SYSTEM_PROMPT,
                "user_prompt": self.USER_PROMPT,
                "temperature": 0.7,
                "max_tokens": max_tok,
                "repeat": 1,
                "latency": round(latency, 3),
                "answer": answer,
                "error": error,
            })

# ========== ЭКСПЕРИМЕНТ 4: ИТОГОВАЯ КОНФИГУРАЦИЯ ==========
class FinalConfigurationExperiment(BaseExperiment):
    name = "final_config"
    description = "Итоговая конфигурация для сценария 'Первичный анализ события ИБ'"

    USER_PROMPT = (
        "Событие: в 02:45 зафиксировано 40 неудачных попыток входа "
        "в учётную запись 'root' с IP-адреса 192.168.1.105, "
        "после чего в 02:47 произошёл успешный вход. "
        "Проанализируй ситуацию."
    )
    SYSTEM_PROMPT = (
        "Ты ассистент аналитика ИБ. Не выдумывай недостающие данные. "
        "Явно отделяй факты от предположений. Ответ верни в формате: "
        "Факты / Гипотезы / Действия."
    )
    TEMPERATURE = 0.3
    MAX_TOKENS = 1000
    REPEATS = 3

    def run(self):
        print(f"\n{'=' * 60}")
        print(f"ЭКСПЕРИМЕНТ 4: {self.description}")
        print(f"Параметры: temp={self.TEMPERATURE}, max_tokens={self.MAX_TOKENS}")
        print(f"Повторов: {self.REPEATS}")
        print('=' * 60)

        for repeat in range(1, self.REPEATS + 1):
            answer, latency, error = self.ask_model(
                prompt=self.USER_PROMPT,
                system=self.SYSTEM_PROMPT,
                temperature=self.TEMPERATURE,
                max_tokens=self.MAX_TOKENS,
            )
            if error:
                print(f"  [{repeat}/{self.REPEATS}] ОШИБКА: {error}")
            else:
                print(f"  [{repeat}/{self.REPEATS}] OK ({latency:.2f} с), "
                      f"длина: {len(answer)} симв.")

            self.results.append({
                "experiment": self.name,
                "model": MODEL,
                "system_id": "sys_final",
                "system_text": self.SYSTEM_PROMPT,
                "user_prompt": self.USER_PROMPT,
                "temperature": self.TEMPERATURE,
                "max_tokens": self.MAX_TOKENS,
                "repeat": repeat,
                "latency": round(latency, 3),
                "answer": answer,
                "error": error,
            })

# ========== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ==========
def save_all_results(results, filename=RESULTS_FILE):
    """Сохраняет все результаты в CSV (дописывает в конец)."""
    if not results:
        print("\nНет результатов для сохранения.")
        return
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)
    print(f"\n✓ Сохранено {len(results)} записей в {filename}")


# ========== ГЛАВНЫЙ ЗАПУСК ==========
EXPERIMENTS = {
    "1": SystemPromptExperiment,
    "2": TemperatureExperiment,
    "3": MaxTokensExperiment,
    "4": FinalConfigurationExperiment,
}


def main():
    print(f"Модель: {MODEL}")
    print("\nДоступные эксперименты:")
    for key, cls in EXPERIMENTS.items():
        print(f"  [{key}] {cls.description}")
    print("  [all] Запустить все эксперименты последовательно")

    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
    else:
        choice = input("\nВыберите эксперимент (1/2/3/all): ").strip()

    all_results = []

    if choice.lower() == "all":
        for key in EXPERIMENTS:
            experiment = EXPERIMENTS[key]()
            experiment.run()
            all_results.extend(experiment.get_results())
    elif choice in EXPERIMENTS:
        experiment = EXPERIMENTS[choice]()
        experiment.run()
        all_results.extend(experiment.get_results())
    else:
        print(f"\nНеизвестный эксперимент: {choice}")
        sys.exit(1)

    save_all_results(all_results)


if __name__ == "__main__":
    main()