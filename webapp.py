from flask import Flask, request, send_from_directory
import os
from EndToEnd import run_end_to_end

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
SLICED_DIR = 'sliced_tiles'
DETECTED_DIR = 'detected_tiles'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(SLICED_DIR, exist_ok=True)
os.makedirs(DETECTED_DIR, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return "No file selected. Please go back and select a TIFF file."
        file = request.files['file']
        if file and (file.filename.endswith('.tif') or file.filename.endswith('.tiff')):
            input_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(input_path)
            base_filename = os.path.splitext(file.filename)[0]
            output_tiff_file = os.path.join(OUTPUT_FOLDER, f"{base_filename}_stitched.tif")
            output_shp_file = os.path.join(OUTPUT_FOLDER, f"{base_filename}_detections.shp")

            run_end_to_end(
                original_tiff_path=input_path,
                sliced_dir=SLICED_DIR,
                detected_dir=DETECTED_DIR,
                output_tiff_path=output_tiff_file,
                output_shp_path=output_shp_file
            )

            tiff_dl_name = os.path.basename(output_tiff_file)
            shp_dl_name = os.path.basename(output_shp_file)
            return f"""
            <!doctype html>
            <title>Processing Complete</title>
            <h1>Processing Complete, the result is available at your local outputs folder !</h1>
            <p>Your files have been generated:</p>
            </ul>
            <p>(Note: For the Shapefile to work, you need all its associated files like .shx, .dbf, etc., from the 'outputs' folder on the server)</p>
            <a href="/">Process another file</a>
            """
    return '''
    <!doctype html>
    <title>Upload TIFF</title>
    <h1>Upload a GeoTIFF for Processing</h1>
    <p>This tool will slice the image, run YOLO detection, and generate two outputs:
    <br>1. A stitched GeoTIFF with detection boxes drawn on it.
    <br>2. A Shapefile of the detection boxes.</p>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file accept=".tif,.tiff">
      <input type=submit value=Start Processing>
    </form>
    '''

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
