from flask import Flask,redirect
import problemmodel
import colorsys
import qubo
import math

app = Flask(__name__)


model = []
feature_colors = []
solutions = []

css = """
body {
nbackground-color: black;
kcolor: white;
}
.solution_detail {
    border-collapse: collapse;
    border: 1px solid;
}
.solution_detail td {
    border-top: 1px solid;
    border-bottom: 1px solid;
    border-right: 1px dashed;
    margin: 0;
    text-align: center
}

.schedule_table {
    width: 100%;
    border-collapse: collapse;
    border: 2px solid;
}
.schedule_table td {
    border-top: 2px solid;
    border-bottom: 2px solid;
    border-right: 1px dashed;
    margin: 0;
    text-align: center
}
.schedule_table th {
    border-top: 2px solid;
    border-bottom: 2px solid;
    border-right: 1px dashed;
    margin: 0;
    text-align: center
}

.resource_table {
    width: 70%;
    border: 1px solid;
    border-collapse: collapse;
}
.resource_table div {
    min-width: 100px;
}
.resource_table th {
    border-bottom: 2px solid;
}
.resource_table td {
    _border-top: 1px solid;
    border-bottom: 1px solid;
    margin: 0;
    text-align: center
}

.feature {
    text-align: center;
}
"""
@app.route("/")
def home():
    return """
    <html>
        <head><title>Resource Attribution Problem</title></head>
        <body>
        <h1>Choose a model to start with</h1>
        <div><a href="/model/tiny">Tiny problem</a></div>
        <div><a href="/model/small">Small problem</a></div>
        <div><a href="/model/big">Big problem</a></div>
        <div><a href="/model/verybig">Very big problem</a></div>
        </body>
    </html>
    """

@app.route("/model")
def main():
    # Define the HTML table
    html = """
    <html>
        <head>
            <title>Simple Table</title>
            <style type="text/css">
    """+css+"""
            </style>
        </head>
        <body>
            <h1>Resources</h1>
    """+resource_table()+"""
            <h1>Solutions</h1>
    """+solutions_summary()+solutions_detail()+"""
    <div>
    <ul>
    <li><a href="/solve/exact">Exact solver</a>
    <li><a href="/solve/simulatedannealing">Simulated Annealing</a>
    <li><a href="/solve/qpu">QPU</a>
    </ul></div>
            <h1>Schedule Table</h1>
    """+model_table()+"""
    
    """
    return html

def solutions_summary():
    html = "<table><tr><th>Algorithm</th><th>Time</th><th>Score</th><th>Valid</th></tr>"
    for solution in solutions:
        html += f"<tr><td>{solution.algorithm}</td><td>{solution.time}</td><td>{solution.coverage}</td><td>{solution.valid}</td></tr>"
    html += "</table>"
    return html

def solutions_detail():
    html = ""
    for index,solution in enumerate(solutions):
        html += f"<table id='solution{index}' class='solution_detail'><tr><th>Task</th><th>coverage</th><th>Resources</th><th>Features</th></tr>"
        for i in range(solution.model.nb_tasks()):
            html += f"""<tr>
                <td>Task {i}</td>
                <td>{math.floor(solution.tasks_coverage[i]*100)}%</td>
                <td>"""+"".join(f"<div>{resource}</div>" for resource in solution.task_resources(i))+"""</td>"
                <td>"""+"".join(f"<div>{feature}</div>" for feature in solution.tasks_features_covered[i])+"""</td>"
            """
        html += "</table>"
    return html

def resource_table():
    html = "<table class='resource_table'><tr><th class='resource'>Resource</th>"
    for f in range(model.nb_features()):
        html += f"<th>Feature {f}</th>"
    html += "</tr>"
    for r in range(model.nb_resources()):
        html += f"<tr><td class='resource'>{r}</td>"
        for f in range(model.nb_features()):
            if model.resources[r,f] == 1:
                html += f"<td><div class='feature' style='background-color:{feature_colors[f]}'>{f}</div></td>"
            else:
                html += "<td></td>"
        html += "</td></tr>"
    html += "</table>"
    return html

def model_table():
    html = "<table class='schedule_table' border='1px solid'><tr><th></th>"
    for s in range(model.nb_schedules()):
        html += f"<th>{s}</th>"
    html += "</tr>"
    for i in range(model.nb_tasks()):
        html += f"<tr><td>Task {i}</td>"
        for s in range(model.nb_schedules()):
            html += "<td>"
            if(model.schedules[i,s] == 1):
                for f in model.task_features(i):
                    html += f"<div class='feature' style='background-color:{feature_colors[f]}'>{f}</div>"
            html += "</td>"
        html += "</tr>"
    html += "</table>"
    print(html)
    return html

@app.route("/solve/exact")
def solve_exact():
    global solutions
    solutions.append(qubo.solve_with_exactSolver(model))
    return redirect("/model", code=302)

@app.route("/solve/simulatedannealing")
def solve_simulatedAnnealing():
    global solutions
    solutions.append(qubo.solve_with_simulatedAnnealing(model))
    return redirect("/model", code=302)

@app.route("/solve/qpu")
def solve_qpu():
    global solutions
    solutions.append(qubo.solve_on_dwave(model))
    return redirect("/model", code=302)


@app.route("/model/tiny")
def load_tiny():
    global model
    global feature_colors
    global solutions
    model = problemmodel.tiny_sample_problem()
    solutions = []
    feature_colors = generate_contrasting_colors(model.nb_features())
    # Redirect to the root route "/"
    return redirect("/model", code=302)

@app.route("/model/small")
def load_small():
    global model
    global feature_colors
    global solutions
    model = problemmodel.small_sample_problem()
    solutions = []
    feature_colors = generate_contrasting_colors(model.nb_features())
    # Redirect to the root route "/"
    return redirect("/model", code=302)

@app.route("/model/big")
def load_big():
    global model
    global feature_colors
    global solutions
    model = problemmodel.big_sample_problem()
    solutions = []
    feature_colors = generate_contrasting_colors(model.nb_features())
    # Redirect to the root route "/"
    return redirect("/model", code=302)


def generate_contrasting_colors(n=20):
    colors = []
    for i in range(n):
        # Evenly spaced hues around the color wheel
        hue = i / n
        # Convert from HSL to RGB with maximum saturation and lightness
        rgb = colorsys.hsv_to_rgb(hue, 1, 1)
        # Convert to RGB scale (0-255)
        rgb_html = f"rgb({int(255 * rgb[0])}, {int(255 * rgb[1])}, {int(255 * rgb[2])})"
        #colors.append(tuple(int(255 * x) for x in rgb))
        colors.append(rgb_html)
    return colors

# Generate 20 contrasting RGB colors
load_tiny()

if __name__ == "__main__":
    app.run(debug=True)

