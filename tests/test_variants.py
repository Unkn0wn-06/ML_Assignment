import numpy as np
import pytest

from data_centre_rl.data import TARGET_COLUMNS, make_transition_table
from data_centre_rl.transition_model import VARIANTS, TransitionModel


@pytest.mark.parametrize("variant", VARIANTS)
def test_rollout_matches_batch_and_variant_contract(sample_data, variant):
    table = make_transition_table(sample_data)
    for column in ("Outlet Avg Temp", "Cooling Return Average"):
        table[f"next_{column}"] = table[column] + 1.25
    model = TransitionModel(n_estimators=5, variant=variant).fit(table)
    predicted = model.predict(table)
    for index, row in table.iterrows():
        single = model.predict_one(row)
        np.testing.assert_allclose(
            [single[target] for target in TARGET_COLUMNS], predicted.loc[index], atol=1e-10
        )
    if variant == VARIANTS[1]:
        np.testing.assert_allclose(predicted["Outlet Avg Temp"], table["Outlet Avg Temp"] + 1.25)
    if variant == VARIANTS[2]:
        grid = table.iloc[[0] * 30].copy()
        grid["Cooling Supply Average"] = np.linspace(14, 26, len(grid))
        assert (
            np.diff(model.predict(grid)[["Outlet Avg Temp", "Cooling Return Average"]], axis=0)
            >= -1e-12
        ).all()
