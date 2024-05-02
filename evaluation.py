"""
The following is a simple example evaluation method.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the evaluation, reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
import json
from glob import glob
import SimpleITK
import random
from multiprocessing import Pool
from statistics import mean
from pathlib import Path
from pprint import pformat, pprint
import os
from picai_eval import evaluate
from picai_eval.data_utils import sterilize


INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")
GROUND_TRUTH_DIRECTORY = Path("ground_truth")
# INPUT_DIRECTORY = Path("/home/pidgeotto/natalia/PANORAMA_evaluation/test/input")
# OUTPUT_DIRECTORY = Path("/home/pidgeotto/natalia/PANORAMA_evaluation/test/output")
# GROUND_TRUTH_DIRECTORY = Path("/home/pidgeotto/natalia/PANORAMA_evaluation/ground_truth")


def main():
    print_inputs()

    metrics = {}
    predictions = read_predictions()

    # We now process each algorithm job for this submission
    # Note that the jobs are not in any order!
    # We work that out from predictions.json

    # Start a number of process workers, using multiprocessing
    # The optimal number of workers ultimately depends on how many
    # resources each process() would call upon

    # with Pool(processes=4) as pool:
    #     metrics["results"] = pool.map(process, predictions)

    # Now generate an overall score(s) for this submission
    # metrics["aggregates"] = {
    #     "my_metric": mean(result["my_metric"] for result in metrics["results"])
    # }

    metrics["aggregates"] = panorama_process(predictions)

    # Make sure to save the metrics
    write_metrics(metrics=metrics)

    return 0


def panorama_process(predictions):
    case_pred = {}
    pred_paths = []
    gt_paths = []
    subject_list = []

    for job in predictions:

        location_pdac_likelihood = get_file_location(
            job_pk=job["pk"],
            values=job["outputs"],
            slug="pdac-likelihood",
        )

        location_pdac_detection_map = get_file_location(
                job_pk=job["pk"],
                values=job["outputs"],
                slug="pdac-detection-map",
        )
        pdac_detection_map_file = glob(str(location_pdac_detection_map / "*.mha"))[0]

        image_name_venous_phase_ct_scan = get_image_name(
            values=job["inputs"],
            slug="venous-phase-ct-scan",
        )

        result_pdac_likelihood = load_json_file(location=location_pdac_likelihood)

        subject_id = image_name_venous_phase_ct_scan.split('_0000.mha')[0]

        ground_trut_path = str(GROUND_TRUTH_DIRECTORY / (subject_id + '.nii.gz'))

        print('subject_id', subject_id)
        print('location_pdac_likelihood', location_pdac_likelihood)
        print('result_pdac_likelihood', result_pdac_likelihood)
        print('location_pdac_detection_map', pdac_detection_map_file)


        gt_paths += [ground_trut_path]
        pred_paths += [pdac_detection_map_file]
        subject_list += [subject_id]
        case_pred[subject_id] = result_pdac_likelihood

    print('Performing evaluation')
    print('pred_paths', pred_paths)
    print('gt_paths', gt_paths)
    print('subject_list', subject_list)

    # perform evaluation
    metrics = evaluate(
        y_det=pred_paths,
        y_true=gt_paths,
        subject_list=subject_list,
        y_true_postprocess_func=lambda lbl: (lbl == 1).astype(int)
    )
    print('Computing Metrics')
    # overwrite default case-level prediction derivation with user-defined one
    metrics.case_pred = case_pred
    print('Metrics done')
    print(metrics.case_pred)

    # store metrics (and add to_dict() conversion to metrics)
    metrics.to_dict = lambda: sterilize(metrics.minimal_dict())
    #self._case_results = metrics
    aggregate_results = {
        "score": 0.1,
        "auroc": 0.1,
        "AP": 0.1,
        "lesion_TPR_at_FPR_0.1": 0.1,
        "lesion_TPR_at_FPR_0.2": 0.1,
        "lesion_TPR_at_FPR_0.3": 0.1,
        "lesion_TPR_at_FPR_0.4": 0.1,
        "lesion_TPR_at_FPR_0.5": 0.1,
        "lesion_TPR_at_FPR_1.0": 0.1,
        "lesion_TPR_at_FPR_5.0": 0.1,
        "num_cases": metrics.num_cases,
        "num_lesions": metrics.num_lesions,
        "picai_eval_version": metrics.version,
    }
    print("Evaluation succeeded.")

    return aggregate_results


def print_inputs():
    # Just for convenience, in the logs you can then see what files you have to work with
    input_files = [str(x) for x in Path(INPUT_DIRECTORY).rglob("*") if x.is_file()]

    print("Input Files:")
    pprint(input_files)
    print("")


def read_predictions():
    # The prediction file tells us the location of the users' predictions
    with open(INPUT_DIRECTORY / "predictions.json") as f:
        return json.loads(f.read())


def get_image_name(*, values, slug):
    # This tells us the user-provided name of the input or output image
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["image"]["name"]

    raise RuntimeError(f"Image with interface {slug} not found!")


def get_interface_relative_path(*, values, slug):
    # Gets the location of the interface relative to the input or output
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["interface"]["relative_path"]

    raise RuntimeError(f"Value with interface {slug} not found!")


def get_file_location(*, job_pk, values, slug):
    # Where a job's output file will be located in the evaluation container
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return INPUT_DIRECTORY / job_pk / "output" / relative_path


def load_json_file(*, location):
    # Reads a json file
    with open(location) as f:
        return json.loads(f.read())


def load_image_file(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def write_metrics(*, metrics):
    # Write a json document used for ranking results on the leaderboard
    with open(OUTPUT_DIRECTORY / "metrics.json", "w") as f:
        f.write(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    raise SystemExit(main())