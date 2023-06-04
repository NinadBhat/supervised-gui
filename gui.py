from IPython.display import display
from ipywidgets import VBox, HBox, Dropdown, BoundedFloatText, Button, Output
from joblib import load
from tabulate import tabulate
import pandas as pd
import pickle
from ipywidgets import HTML


FEATURE_COLUMNS = [
    "Processing", "Ag", "Al", "B", "Bi", "Be", "Cd", "Co", "Cr", "Cu", "Er", "Eu",
    "Fe", "Ga", "Li", "Mg", "Mn", "Ni", "Sc", "Si", "Sn", "Ti", "V", "Zn", "Zr"
]

PROCESSES_ENCODING = {
    "as-cast or as-fabricated": "No Processing",
    "Annealed, Solutionised": "Solutionised",
    "H (soft)": "Strain hardened",
    "H (hard)": "Strain Harderned (Hard)",
    "T1": "Naturally aged",
    "T3 (incl. T3xx)": "Solutionised + Cold Worked + Naturally aged",
    "T4": "Solutionised + Naturally aged",
    "T5": "Artificial aged",
    "T6 (incl. T6xx)": "Solutionised  + Artificially peak aged",
    "T7 (incl. T7xx)": "Solutionised + Artificially over aged",
    "T8 (incl. T8xx)": "Solutionised + Artificially over aged",
}


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def calculate_mechanical_properties(feature_dict):
    model = load_pickle("models/rf_model.pkl")
    preprocessor = load_pickle("models/preprocessor.pkl")

    feature_df = pd.DataFrame(feature_dict, index=[0])
    processed_feature = preprocessor.transform(feature_df)
    properties = model.predict(processed_feature)

    return properties[0]



def print_properties(concentration_widgets, output, process_type):
    with output:
        concentrations = [widget.value / 100 for widget in concentration_widgets]
        process = PROCESSES_ENCODING[process_type.value]
        concentrations.insert(1, 1 - sum(concentrations))
        input_dict = dict(zip(FEATURE_COLUMNS, [process] + concentrations))

        print_conc_list = [[element, f"{conc*100:.2f}"] for element, conc in input_dict.items() if element != "Processing" and conc > 1e-06]

        properties = calculate_mechanical_properties(input_dict)

        # Create HTML table rows manually for concentrations
        conc_rows = "".join([f"<tr><td style='text-align:left;'>{element}</td><td style='text-align:right;'>{conc}</td></tr>" for element, conc in print_conc_list])

        display(HTML(
            f"""
            <div style='border:2px solid #000; padding:20px; margin-top:20px;'>
                <h2 style='color: #005599; '>Element Concentrations (wt%):</h2>
                <table style='width:100%; border-collapse: collapse;'>
                    {conc_rows}
                </table>
            </div>
            """
        ))

        display(HTML(
            f"""
            <div style='border:2px solid #000; padding:20px; margin-top:20px;'>
                <h2 style='color: #005599; '>Mechanical Properties:</h2>
                <table style='width:100%; border-collapse: collapse;'>
                    <tr>
                        <th style='text-align:left;'>Property</th>
                        <th style='text-align:right;'>Value</th>
                    </tr>
                    <tr>
                        <td style='text-align:left;'>Yield Strength (MPa)</td>
                        <td style='text-align:right; color: #005599;'>{properties[0]:.2f}</td>
                    </tr>
                    <tr>
                        <td style='text-align:left;'>Tensile Strength (MPa)</td>
                        <td style='text-align:right; color: #005599;'>{properties[1]:.2f}</td>
                    </tr>
                    <tr>
                        <td style='text-align:left;'>Elongation (%)</td>
                        <td style='text-align:right; color: #005599;'>{properties[2]:.2f}</td>
                    </tr>
                </table>
            </div>
            """
        ))


def build_concentration_widget(element):
    return BoundedFloatText(
        value=0,
        min=0,
        max=100.0,
        step=0.1,
        description=f"{element}:",
        disabled=False,
    )


def build_gui():
    concentration_elements = [element for element in FEATURE_COLUMNS if element != "Processing"]
    concentration_widgets = [build_concentration_widget(element) for element in concentration_elements]

    process_type = Dropdown(
        options=list(PROCESSES_ENCODING.keys()),
        value="as-cast or as-fabricated",
        description="Process:",
        disabled=False,
    )

    button = Button(description="Calculate Properties")
    output = Output()
    button.on_click(lambda b: print_properties(concentration_widgets, output, process_type))

    display(VBox([process_type]))

    left_widgets = VBox(concentration_widgets[:len(concentration_widgets) // 2])
    right_widgets = VBox(concentration_widgets[len(concentration_widgets) // 2:])

    display(HBox([left_widgets, right_widgets]))

    display(VBox([button, output]))
