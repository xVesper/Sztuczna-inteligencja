import gradio as gr
import pandas as pd


def read_table(file_name):
    try:
        table = pd.read_csv(file_name)
        return table
    except Exception as e:
        return None


def display_table(table, num_rows):
    if table is None:
        return "Nie udało się wczytać pliku. Sprawdź nazwę i format pliku."
    elif num_rows == 0:
        return "Musisz podać wiersze"
    else:
        num_rows = min(num_rows, len(table))
        return str(table.head(num_rows))


def get_table_info(table):
    if table is None:
        return "Nie udało się wczytać pliku. Sprawdź nazwę i format pliku."

    num_objects = len(table)  #liczba obiektow czyli suma komorek tutaj 10k
    num_attributes = len(table.columns) -1 #liczba atrybutow to ilosc kolumn
    num_classes = len(table[table.columns[-1]].unique()) #liczba klas to unikalne wartosci w kolumnie
    class_sizes = table[table.columns[-1]].value_counts().to_dict() #ilosc wystapien danej klasy np klasa 1 to 700x wystapien itp

    info_str = f"Tabela zawiera {num_objects} obiektów i {num_attributes} atrybutów.\n"
    info_str += f"Liczba klas decyzyjnych: {num_classes}\n"
    info_str += "Wielkość każdej klasy decyzyjnej:\n"
    for cls, size in class_sizes.items():
        info_str += f" - {cls}: {size}\n"

    return info_str


def predict(file_name, num_rows):
    table = read_table(file_name)
    table_display = display_table(table, num_rows)
    table_info = get_table_info(table)

    return f"{table_display}\n\n{table_info}"


iface = gr.Interface(fn=predict,
                     inputs=["text", "slider"],
                     outputs="text",
                     examples=[["test.csv", 5]],
                     title="Chatbot dla tabel decyzyjnych")
iface.launch()