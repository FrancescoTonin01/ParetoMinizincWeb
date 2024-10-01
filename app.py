from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import os
import sys
import io
import matplotlib.pyplot as plt
import base64
from paretoV2 import main as pareto_main

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route('/solve', methods=['POST', 'OPTIONS'])
def solve():
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
    else:
        input_type = request.form.get('inputType')
        timeout = int(request.form.get('timeout', 60))
        temp_mzn_path = None
        temp_dzn_path = None

        try:
            if input_type == 'file':
                minzinc_file = request.files.get('minzincFile')
                datazinc_file = request.files.get('datazincFile')

                if minzinc_file:
                    temp_mzn = tempfile.NamedTemporaryFile(mode='w+', suffix='.mzn', delete=False)
                    minzinc_file.save(temp_mzn.name)
                    temp_mzn_path = temp_mzn.name
                    temp_mzn.close()

                if datazinc_file:
                    temp_dzn = tempfile.NamedTemporaryFile(mode='w+', suffix='.dzn', delete=False)
                    datazinc_file.save(temp_dzn.name)
                    temp_dzn_path = temp_dzn.name
                    temp_dzn.close()
            else:
                minzinc_text = request.form.get('minzincText')
                datazinc_text = request.form.get('datazincText')

                if minzinc_text:
                    temp_mzn = tempfile.NamedTemporaryFile(mode='w+', suffix='.mzn', delete=False)
                    temp_mzn.write(minzinc_text)
                    temp_mzn_path = temp_mzn.name
                    temp_mzn.close()

                if datazinc_text:
                    temp_dzn = tempfile.NamedTemporaryFile(mode='w+', suffix='.dzn', delete=False)
                    temp_dzn.write(datazinc_text)
                    temp_dzn_path = temp_dzn.name
                    temp_dzn.close()

            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            try:
                pareto_main(temp_mzn_path, temp_dzn_path, solver_type="gecode", timeout=timeout, all_solutions=False)
                output = sys.stdout.getvalue()
                status = 'success'
            except Exception as e:
                output = f"Si è verificato un errore durante l'elaborazione: {str(e)}"
                status = 'error'

            img = None
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                img = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close(fig)
                break

            response = jsonify({'output': output, 'image': img, 'status': status})

        except Exception as e:
            response = jsonify({'error': str(e), 'status': 'error'}), 500

        finally:
            sys.stdout = old_stdout
            plt.close('all')  # Chiudi tutte le figure di Matplotlib
            if temp_mzn_path and os.path.exists(temp_mzn_path):
                try:
                    os.unlink(temp_mzn_path)
                except PermissionError:
                    print(f"Impossibile eliminare il file temporaneo: {temp_mzn_path}")
            if temp_dzn_path and os.path.exists(temp_dzn_path):
                try:
                    os.unlink(temp_dzn_path)
                except PermissionError:
                    print(f"Impossibile eliminare il file temporaneo: {temp_dzn_path}")

    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

if __name__ == '__main__':
    app.run(debug=True)