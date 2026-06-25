"""Изолируем тесты от рабочей БД: отдельный временный файл."""
import os
import tempfile

# должно выполниться ДО импорта app.main (создаёт Database по DB_PATH)
os.environ["DB_PATH"] = os.path.join(tempfile.gettempdir(), "alem_test.db")
