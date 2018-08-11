import pytest
from hypothesis import given, assume
import hypothesis.strategies as st
from hypothesis.extra.pandas import data_frames, column

from gbd_mapping import causes, risks, Cause, Risk, Etiology, Sequela

from vivarium_inputs.utilities import get_age_group_midpoint_from_age_group_id
from vivarium_inputs.data_artifact import _normalize, _dump_dataframe, _dump_json_blob, EntityKey

CAUSE_MEASURES = ["death", "prevalence", "incidence", "population_attributable_fraction",
                  "cause_specific_mortality", "excess_mortality", "remission"]
RISK_MEASURES = ["exposure", "exposure_standard_deviation", "relative_risk"]
POPULATION_ENTITY_PATHS = ["population.structure", "population.age_bins",
                           "population.theoretical_minimum_risk_life_expectancy"]


@st.composite
def measure(draw, entities, measures):
    measure = draw(st.sampled_from(measures))
    entity = draw(st.sampled_from(list(entities)))
    assume(entity is not None)
    entity_type = {
            Cause: "cause",
            Risk: "risk_factor",
            Etiology: "etiology",
            Sequela: "sequela",
    }[type(entity)]
    return f"{entity_type}.{entity.name}.{measure}"


@st.composite
def measure_dataframes(draw):
    columns = draw(st.sets(st.sampled_from([
                                 ("location", str),
                                 ("measure", str),
                                 ("year_id", int),
                                 ("sex_id", int),
                                 ("age_group_id", int)]
    )))
    columns = [column(c[0], dtype=c[1]) for c in columns] + [column(f"draw_{i}", dtype=float) for i in range(10)]
    df = draw(data_frames(columns=columns))
    assume(not df.empty)
    return df


@pytest.mark.skip("Cluster")
@given(measure_dataframes())
def test__normalize(data):
    normed = _normalize(data)

    original_draw_count = len([c for c in data.columns if c.startswith("draw_")])
    normalized_draw_count = len(normed.draw.unique())
    assert original_draw_count == normalized_draw_count

    if "age_group_ids" in data:
        assert "age" in normed
        assert "age_group_start" in normed
        assert "age_group_end" in normed
        no_draw_orig = data[[c for c in data.columns if not c.startswith("draw_")]].unique()
        no_draw_norm = normed[[c for c in normed.columns if c != "draw"]].unique()
        assert no_draw_norm == get_age_group_midpoint_from_age_group_id(no_draw_orig)


@pytest.mark.skip("Cluster")
@given(entity_path=st.one_of(measure(causes, CAUSE_MEASURES), measure(risks, RISK_MEASURES),
                             st.sampled_from(POPULATION_ENTITY_PATHS)),
       columns=st.sets(st.sampled_from(["year", "location", "draw", "cause", "risk"])
                       | st.text(min_size=1, max_size=30)),
       path=st.text(alphabet="abcdefghijklmnopqrstuvwxyz1234567890_/"), )
def test__dump_dataframe(entity_path, columns, path, mocker):
    entity_key = EntityKey(entity_path)

    mock_pd = mocker.patch("vivarium_inputs.data_artifact.pd")
    data = mocker.Mock()
    data.empty = False
    data.columns = list(columns)

    _dump_dataframe(data, entity_key, entity_key.measure, path)

    mock_pd.HDFStore.assert_called_with(path, complevel=mocker.ANY, format="table")

    expected_columns = list({"year", "location", "draw", "cause", "risk"}.intersection(columns))
    mock_pd.HDFStore().__enter__().put.assert_called_with(entity_key.to_path(), data,
                                                          format="table", data_columns=set(expected_columns))


@given(entity_path=st.one_of(measure(causes, CAUSE_MEASURES), measure(risks, RISK_MEASURES),
                             st.sampled_from(POPULATION_ENTITY_PATHS)),
       path=st.text(alphabet="abcdefghijklmnopqrstuvwxyz1234567890_/"))
def test__dump_json_blob(entity_path, path, mocker):
    entity_key = EntityKey(entity_path)

    mock_tables = mocker.patch("vivarium_inputs.data_artifact.tables")
    mock_filenode = mocker.patch("vivarium_inputs.data_artifact.filenode")
    data = {1: 2}

    _dump_json_blob(data, entity_key, entity_key.measure, path)

    mock_tables.open_file.assert_called_with(path, "a")
    mock_tables.open_file().create_group.assert_called_with(entity_key.group_prefix, entity_key.group_name,
                                                            createparents=True)
    mock_filenode.new_node.assert_called_with(mock_tables.open_file(),
                                              where=entity_key.group, name=entity_key.measure)
    mock_filenode.new_node().write.assert_called()  # TODO check data
