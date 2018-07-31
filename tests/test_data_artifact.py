import os

import pytest
from hypothesis import given, assume
import hypothesis.strategies as st
from hypothesis.extra.pandas import data_frames, column

from gbd_mapping import causes, risks, Cause, Risk, Etiology, Sequela

from vivarium_inputs.utilities import get_age_group_midpoint_from_age_group_id
from vivarium_inputs.data_artifact import (_entities_by_type, _normalize, _prepare_key, _parse_entity_path,
                                           _dump_dataframe, _dump_json_blob)

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


@given(st.sets(measure(causes, CAUSE_MEASURES) | measure(risks, RISK_MEASURES)))
def test__entities_by_type(measures):
    print(measures)
    by_type = _entities_by_type(measures)

    types = {m.split(".")[0] for m in measures}

    to_check = {(entity_type, entity_name) for entity_type, entity_names in by_type.items() for entity_name in entity_names}
    for m in measures:
        entity_type, entity_name, _ = m.split(".")
        assert entity_type in by_type
        assert entity_name in by_type[entity_type]
        if (entity_type, entity_name) in to_check:
            to_check.remove((entity_type, entity_name))
    assert not to_check


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


@given(st.one_of(measure(causes, CAUSE_MEASURES), measure(risks, RISK_MEASURES),
                 st.sampled_from(POPULATION_ENTITY_PATHS)))
def test__parse_entity_path(entity_path):
    entity_type, entity_name, measure = _parse_entity_path(entity_path)

    chunks = entity_path.split(".")

    if len(chunks) == 3:
        assert entity_type == chunks[0]
        assert entity_name == chunks[1]
        assert measure == chunks[2]
    else:
        assert entity_type == chunks[0]
        assert entity_name is None
        assert measure == chunks[1]


@given(st.one_of(measure(causes, CAUSE_MEASURES), measure(risks, RISK_MEASURES),
                 st.sampled_from(POPULATION_ENTITY_PATHS)))
def test__prepare_key(entity_path):
    entity_type, entity_name, measure = _parse_entity_path(entity_path)

    key_components = _prepare_key(entity_type, entity_name, measure)

    expected_length = 4
    assert key_components[0] == "/"
    if entity_name is None:
        expected_length -= 1

    assert key_components[1] == entity_type
    assert key_components[-1] == measure
    assert len(key_components) == expected_length


@pytest.mark.skip("Cluster")
@given(entity_path=st.one_of(measure(causes, CAUSE_MEASURES), measure(risks, RISK_MEASURES),
                             st.sampled_from(POPULATION_ENTITY_PATHS)),
       columns=st.sets(st.sampled_from(["year", "location", "draw", "cause", "risk"])
                       | st.text(min_size=1, max_size=30)),
       path=st.text(alphabet="abcdefghijklmnopqrstuvwxyz1234567890_/"), )
def test__dump_dataframe(entity_path, columns, path, mocker):
    key_components = _prepare_key(*_parse_entity_path(entity_path))

    mock_pd = mocker.patch("vivarium_inputs.data_artifact.pd")
    data = mocker.Mock()
    data.empty = False
    data.columns = list(columns)

    _dump_dataframe(data, key_components, path)

    mock_pd.HDFStore.assert_called_with(path, complevel=mocker.ANY, format="table")

    expected_columns = list({"year", "location", "draw", "cause", "risk"}.intersection(columns))
    mock_pd.HDFStore().__enter__().put.assert_called_with(os.path.join(*key_components), data,
                                                          format="table", data_columns=set(expected_columns))


@given(entity_path=st.one_of(measure(causes, CAUSE_MEASURES), measure(risks, RISK_MEASURES),
                             st.sampled_from(POPULATION_ENTITY_PATHS)),
       path=st.text(alphabet="abcdefghijklmnopqrstuvwxyz1234567890_/"))
def test__dump_json_blob(entity_path, path, mocker):
    key_components = _prepare_key(*_parse_entity_path(entity_path))

    mock_tables = mocker.patch("vivarium_inputs.data_artifact.tables")
    mock_filenode = mocker.patch("vivarium_inputs.data_artifact.filenode")
    data = {1: 2}

    _dump_json_blob(data, key_components, path)

    mock_tables.open_file.assert_called_with(path, "a")
    mock_tables.open_file().create_group.assert_called_with(os.path.join(*key_components[:-2]),
                                                            key_components[-2], createparents=True)
    mock_filenode.new_node.assert_called_with(mock_tables.open_file(),
                                              where=os.path.join(*key_components[:-1]), name=key_components[-1])
    mock_filenode.new_node().write.assert_called()  # TODO check data
