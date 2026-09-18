import pytest
from nrr.compiler import graph_evidence, require_external_baseline

def observed(names):
    return {'status': 'observed', 'input_sha256': 'a'*64, 'command': ['vela','model.tflite'],
            'host_operator_count': 2, 'npu_operator_count': 1,
            'operators': [{'device':'host','output_names':names}]}

def test_region_names_are_path_tokens_not_substrings():
    with pytest.raises(ValueError, match='attribute'):
        require_external_baseline(observed(['r01/norm','error1/op']))

def test_exact_region_names_match_tensor_paths():
    require_external_baseline(observed(['StatefulPartitionedCall/r0/norm/mul;r1/norm/add']))

def test_compiled_custom_operator_count_is_labelled_as_partitions():
    r = graph_evidence([{'id':0,'op':'ethos-u','device':'npu','inputs':[0],'outputs':[1]}])
    assert r.get('npu_partition_count') == 1
    assert r.get('count_scope') == 'post_compilation_graph_not_original_model_ops'

def test_non_topological_or_cyclic_graph_is_rejected():
    nodes=[{'id':0,'op':'a','device':'host','inputs':[2],'outputs':[1]},
           {'id':1,'op':'b','device':'host','inputs':[1],'outputs':[2]}]
    with pytest.raises(ValueError, match='topological'):
        graph_evidence(nodes)
