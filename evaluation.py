import json
from pathlib import Path
from typing import Any, Dict, List

from evalutils import ClassificationEvaluation
from evalutils.io import SimpleITKLoader
from evalutils.validators import NumberOfCasesValidator
from picai_eval import evaluate
from picai_eval.data_utils import sterilize


def load_predictions_json(path: Path) -> Dict[str, str]:
    """
    Map prediction filenames to original filenames
    See https://grand-challenge.org/documentation/automated-evaluation-2/
    """
    detection_map_pk_to_subject_id = {}

    with open(path, "r") as f:
        entries = json.load(f)

    if isinstance(entries, float):
        raise TypeError(f"entries of type float for file: {path}")

    for entry in entries:
        # Find case name through input file name
        subject_id = None
        for input in entry["inputs"]:
            if input["interface"]["slug"] == "transverse-t2-prostate-mri":
                filename = str(input["image"]["name"])
                subject_id = filename.split("_t2w")[0]
                break
        if subject_id is None:
            raise ValueError(f"No filename found for entry: {entry}")

        # Find output value for this case
        for output in entry["outputs"]:
            if output["interface"]["slug"] == "cspca-detection-map":
                pk = str(output["image"]["pk"])
                if not pk.endswith(".mha"):
                    pk += ".mha"
                detection_map_pk_to_subject_id[pk] = subject_id

    return detection_map_pk_to_subject_id


class PICAILoader(SimpleITKLoader):
    """
    Custom file loader for PI-CAI input/output interfaces.
    Skips directories and JSON files during prediction exploration.
    Otherwise, it continues as usual.
    """

    def load(self, *, fname: Path) -> List[Dict[str, Any]]:
        if fname.is_dir() or fname.suffix == ".json":
            # skip directories and JSON files
            return []

        return super().load(fname=fname)

    @staticmethod
    def load_case_pred(fname: Path) -> float:
        with open(fname) as fp:
            return float(json.load(fp))


class picai_gc_evaluator(ClassificationEvaluation):
    def __init__(self):
        super().__init__(
            file_loader=PICAILoader(),
            validators=[
                NumberOfCasesValidator(num_cases=100),
            ],
            file_sorter_key=lambda fname: fname.stem,
        )

    def score(self) -> None:
        """
        The evaluation of PI-CAI leverages the picai_eval repository, which implements
        a different way to aggregate than evalutils. Therefore, we override `score`
        instead of `score_case` and `score_aggregates`.
        """
        # collect list of annotation paths and prediction paths
        gt_paths, pred_paths, subject_list, case_pred = [], [], [], {}
        for (_, gt_row), (_, pred_row) in zip(
            self._ground_truth_cases.iterrows(),
            self._predictions_cases.iterrows()
        ):
            # check DataFrames are aligned correctly
            assert pred_row.ground_truth_path == gt_row.path, \
                f"Order mismatch gt & pred DataFrames! {pred_row.ground_truth_path} != {gt_row.path}"

            # Check that they're the right images
            if (self._file_loader.hash_image(self._file_loader.load_image(gt_row.path)) != gt_row["hash"] or
                    self._file_loader.hash_image(self._file_loader.load_image(pred_row.path)) != pred_row["hash"]):
                raise RuntimeError("Images do not match")

            # grab paths and store
            gt_paths += [gt_row.path]
            pred_paths += [pred_row.path]
            subject_list += [pred_row.subject_id]

            # collect case-level csPCa predictions
            case_pred[pred_row.subject_id] = pred_row.cspca_case_level_likelihood

        # perform evaluation
        metrics = evaluate(
            y_det=pred_paths,
            y_true=gt_paths,
            subject_list=subject_list,
        )

        # overwrite default case-level prediction derivation with user-defined one
        metrics.case_pred = case_pred

        # store metrics (and add to_dict() conversion to metrics)
        metrics.to_dict = lambda: sterilize(metrics.minimal_dict())
        self._case_results = metrics
        self._aggregate_results = {
            "score": metrics.score,
            "auroc": metrics.auroc,
            "AP": metrics.AP,
            "lesion_TPR_at_FPR_0.1": float(metrics.lesion_TPR_at_FPR(0.1)),
            "lesion_TPR_at_FPR_0.2": float(metrics.lesion_TPR_at_FPR(0.2)),
            "lesion_TPR_at_FPR_0.3": float(metrics.lesion_TPR_at_FPR(0.3)),
            "lesion_TPR_at_FPR_0.4": float(metrics.lesion_TPR_at_FPR(0.4)),
            "lesion_TPR_at_FPR_0.5": float(metrics.lesion_TPR_at_FPR(0.5)),
            "lesion_TPR_at_FPR_1.0": float(metrics.lesion_TPR_at_FPR(1.0)),
            "lesion_TPR_at_FPR_5.0": float(metrics.lesion_TPR_at_FPR(5.0)),
            "num_cases": metrics.num_cases,
            "num_lesions": metrics.num_lesions,
            "picai_eval_version": metrics.version,
        }
        print("Evaluation succeeded.")

    def load(self):
        """Define custom load function for T2 algorithm submissions"""
        # read which predictions and ground truths are available
        self._ground_truth_cases = self._load_cases(
            folder=self._ground_truth_path)
        self._predictions_cases = self._load_cases(
            folder=self._predictions_path)

        # parse predictions.json to create mapping between predictions and gt
        self.mapping_dict = load_predictions_json(
            Path("/input/predictions.json"))

        # set subject_id of predictions
        self._predictions_cases["subject_id"] = self._predictions_cases.apply(
            lambda row: self.mapping_dict[Path(row.path).name],
            axis=1
        )

        # set case-level predictions
        self._predictions_cases["cspca_case_level_likelihood"] = self._predictions_cases.apply(
            lambda row: self._file_loader.load_case_pred(
                Path(row.path).parent.parent.parent / "cspca-case-level-likelihood.json"
            ),
            axis=1
        )

        # set corresponding ground truth path
        self._predictions_cases["ground_truth_path"] = self._predictions_cases.apply(
            lambda row: self._ground_truth_path / (row.subject_id + ".nii.gz"),
            axis=1
        )

        # sort predictions and annotations DataFrames in the same way
        self._ground_truth_cases = self._ground_truth_cases.sort_values(
            "path"
        ).reset_index(drop=True)
        self._predictions_cases = self._predictions_cases.sort_values(
            "ground_truth_path"
        ).reset_index(drop=True)


if __name__ == "__main__":
    picai_gc_evaluator().evaluate()