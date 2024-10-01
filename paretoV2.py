import argparse
import asyncio
from enum import Enum
from typing import AsyncIterator, List, Tuple
from minizinc import Instance, Result, Model, Solver
import matplotlib.pyplot as plt
import re
import tempfile
import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')  # Usa un backend non interattivo

class OptDirection(Enum):
    MAXIMIZE = 1
    MINIMIZE = 2

    def cmp_op(self) -> str:
        if self == OptDirection.MAXIMIZE:
            return ">"
        elif self == OptDirection.MINIMIZE:
            return "<"
        else:
            raise ValueError("Invalid optimization direction")

    def better(self, a, b) -> bool:
        if self == OptDirection.MAXIMIZE:
            return a > b
        elif self == OptDirection.MINIMIZE:
            return a < b
        else:
            raise ValueError("Invalid optimization direction")


async def pareto_solutions(
    inst: Instance, objectives: List[Tuple[str, OptDirection]], *args, **kwargs
) -> AsyncIterator[Result]:
    with inst.branch() as branch:
        result = await branch.solve_async(*args, **kwargs)
        while result.status.has_solution():
            yield result
            branch.add_string(
                "constraint "
                + "\\/".join(
                    [
                        f"({name} {o.cmp_op()} {result[name]})"
                        for (name, o) in objectives
                    ]
                )
                + ";"
            )
            result = await branch.solve_async(*args, **kwargs)


async def run_pareto_with_timeout(instance, objectives, timeout: int):
    partial_results = []  # Inizializza una lista per risultati parziali
    try:
        # Definiamo una task per eseguire il ciclo asincrono
        async def collect_solutions():
            async for result in pareto_solutions(instance, objectives):
                partial_results.append(result)
        
        # Ora applichiamo il timeout all'intero ciclo
        await asyncio.wait_for(collect_solutions(), timeout=timeout)
    
    except asyncio.TimeoutError:
        # Gestione del timeout
        print(f"Timeout di {timeout} secondi raggiunto. Restituisco le soluzioni trovate fino ad ora.")
    except Exception as e:
        # Gestione di eventuali errori
        print(f"Errore durante l'esecuzione: {str(e)}")

  
    # Restituiamo i risultati filtrati
    return partial_results

# Funzione per trovare il pareto front
async def pareto_front(
    inst: Instance, objectives: List[Tuple[str, OptDirection]], *args, **kwargs
) -> List[Result]:
    solns = []  # Lista di soluzioni
    async for res in pareto_solutions(inst, objectives, *args, **kwargs):
        is_dominated = False
        solns_to_remove = []
        for existing_sol in solns:
            if all(o.better(res[name], existing_sol[name]) or res[name] == existing_sol[name] for name, o in objectives) and \
               any(o.better(res[name], existing_sol[name]) for name, o in objectives):
                solns_to_remove.append(existing_sol)
            elif all(o.better(existing_sol[name], res[name]) or existing_sol[name] == res[name] for name, o in objectives) and \
                 any(o.better(existing_sol[name], res[name]) for name, o in objectives):
                is_dominated = True
                break
        if not is_dominated:
            solns = [sol for sol in solns if sol not in solns_to_remove]
            solns.append(res)
    return solns  # Ritorna le soluzioni finali (complete o parziali in caso di timeout)

def pareto_front_sync(results: List[Result], objectives: List[Tuple[str, OptDirection]]) -> List[Result]:
    solns = []  # Lista di soluzioni finali
    for res in results:
        is_dominated = False
        solns_to_remove = []
        for existing_sol in solns:
            if all(o.better(res[name], existing_sol[name]) or res[name] == existing_sol[name] for name, o in objectives) and \
               any(o.better(res[name], existing_sol[name]) for name, o in objectives):
                solns_to_remove.append(existing_sol)
            elif all(o.better(existing_sol[name], res[name]) or existing_sol[name] == res[name] for name, o in objectives) and \
                 any(o.better(existing_sol[name], res[name]) for name, o in objectives):
                is_dominated = True
                break
        if not is_dominated:
            solns = [sol for sol in solns if sol not in solns_to_remove]
            solns.append(res)
    return solns  # Ritorna le soluzioni finali


def extract_and_remove_solve_statement(file_path: str):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    new_lines = []
    objectives = []

    for line in lines:
        if line.startswith("solve"):
            # Check the number of square brackets
            if line.count('[') == 2:
                onearray = True
            elif line.count('[') == 4:
                twoarrays = True

            # Modify the regular expression
            matches = re.findall(r'(maximize|minimize)\s+(\w+(?:\[\d+\])?)', line)

            if matches:
                objectives = [
                    (var, OptDirection.MAXIMIZE if direction == 'maximize' else OptDirection.MINIMIZE)
                    for direction, var in matches
                ]
        else:
            new_lines.append(line)

    # Check if there are variables with "[]" in the objectives
    array_vars = [var for var, _ in objectives]
    helper = ["bbb", "ccc"]
    renamed_objectives = []
    
    for h, var in zip(helper, array_vars):
        # Add the new variable to renamed_objectives
        direction = next(dir for v, dir in objectives if v == var)  # Get the direction of the objective
        renamed_objectives.append((h, direction))  # Add the association (helper, direction)

        # Add variable declarations and constraints to new_lines
        new_lines.append(f'var int: {h};\n')  # Add the declaration of the variable
        new_lines.append(f'constraint {h} = {var};\n')  # Associate the helper element with the variable with "[]"

    objectives = renamed_objectives  # Update objectives with the new names

    # Create the temporary file
    temp_model_file = file_path.replace(".mzn", "_temp.mzn")
    with open(temp_model_file, 'w') as temp_file:
        temp_file.writelines(new_lines)

    return objectives, temp_model_file, array_vars


def main(model_file: str, data_file: str = None, solver_type: str = "gecode", timeout: int = None, all_solutions: bool = False):
    variables, temp_model_file, ogvar = extract_and_remove_solve_statement(model_file)
    model = Model(temp_model_file)
    
    if data_file:
        model.add_file(data_file)

    solver = Solver.lookup(solver_type)
    instance = Instance(solver, model)

    if not variables:
        print("No variables specified for optimization. Simply solving the model...")

        command = ["minizinc", temp_model_file]
        if data_file:
            command.append(data_file)

        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Error during MiniZinc execution:")
            print(result.stderr)
        os.remove(temp_model_file)
        return

    print("Finding solutions...")

    # Invece di usare un loop esplicito, usa direttamente asyncio.run()
    if timeout:
        results = asyncio.run(run_pareto_with_timeout(instance, variables, timeout))
    else:
        results = asyncio.run(pareto_front(instance, variables))

    # Se l'utente ha richiesto tutte le soluzioni, mostra tutte quelle trovate
    if all_solutions:
        filtered_results = results  # Mostra tutte le soluzioni senza filtro
    else:
        filtered_results = pareto_front_sync(results, variables)  # Filtra solo quelle non dominate

    # Mostra le soluzioni trovate
    x_var, y_var = ogvar[0], ogvar[1] if len(ogvar) > 1 else None
    x_values = [res[variables[0][0]] for res in filtered_results]
    y_values = [res[variables[1][0]] for res in filtered_results] if y_var else [0] * len(x_values)
    
    for res in filtered_results:
        print(", ".join(f"{original_var}: {res[helper_var]}" for original_var, (helper_var, _) in zip(ogvar, variables)))

    if variables:
        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, y_values, color='blue', s=100)
    
        for x, y in zip(x_values, y_values):
            plt.annotate(f'({x}, {y})', (x, y), xytext=(5, 5), textcoords='offset points')
    
        plt.xlabel(x_var)
        plt.ylabel(y_var if y_var else "")
        plt.title(f'Pareto Front: {x_var} vs {y_var}' if y_var else f'Pareto Front: {x_var}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    os.remove(temp_model_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to process a MiniZinc model and find the Pareto front of solutions.",
        epilog="Example usage:\n"
               "  python script.py model.mzn --timeout 60 --all-solutions \n"
               "  python script.py model.mzn --data_file data.dzn --solver chuffed",
                formatter_class=argparse.RawTextHelpFormatter  # Questo forza i ritorni a capo

    )

    parser.add_argument(
        "model_file", 
        type=str, 
        help="Path to the MiniZinc model file (.mzn)"
    )

    parser.add_argument(
        "data_file",  # Mantieni come argomento posizionale
        nargs='?',  # Rende l'argomento opzionale
        type=str, 
        default=None, 
        help="Optional path to the MiniZinc data file (.dzn)."
    )

    parser.add_argument(
        "--solver", 
        type=str, 
        default="gecode", 
        help="Solver to use for MiniZinc (default: gecode). Other options might include 'chuffed', 'osicbc', etc."
    )

    parser.add_argument(
        "--timeout", 
        type=int, 
        default=None, 
        help="Timeout in seconds for the optimization process. If not specified, no timeout will be applied."
    )

    parser.add_argument(
        "--all-solutions", 
        action="store_true", 
        help="If specified, displays all solutions found (including dominated ones). By default, only non-dominated solutions (Pareto front) are shown."
    )

    # Parse arguments
    args = parser.parse_args()

    # Call main with parsed arguments
    main(args.model_file, args.data_file, args.solver, args.timeout, args.all_solutions)
