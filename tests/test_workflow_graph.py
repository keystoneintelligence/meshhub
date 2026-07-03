from models.workflow_graph import WorkflowGraph, WorkflowStageStatus


def test_text_generation_workflow_builds_expected_stage_order():
    graph = WorkflowGraph.for_generation_request(
        workflow_id="wf-1",
        mode="Text to 3D",
        model="mesh-provider",
        requested_faces=4096,
        output_folder="runs/one",
        text_prompt="a small crate",
        image_path=None,
        texture_model="texture-provider",
        seed=99,
    )

    assert [stage.key for stage in graph.stages] == [
        "prompt",
        "image_candidates",
        "selected_image",
        "mesh",
        "cleanup",
        "texture",
        "export",
    ]
    assert graph.stage("prompt").status is WorkflowStageStatus.READY
    assert graph.stage("mesh").settings == {
        "model": "mesh-provider",
        "target_faces": 4096,
        "seed": 99,
    }
    assert graph.stage("texture").settings == {"texture_model": "texture-provider"}
    assert graph.stage("export").settings == {"output_folder": "runs/one"}


def test_image_generation_workflow_starts_from_source_image():
    graph = WorkflowGraph.for_generation_request(
        workflow_id="wf-1",
        mode="Image to 3D",
        text_prompt=None,
        image_path="input.png",
        texture_model=None,
    )

    assert [stage.key for stage in graph.stages] == [
        "selected_image",
        "mesh",
        "cleanup",
        "export",
    ]
    assert graph.stage("selected_image").artifacts == ("input.png",)
    assert graph.stage("export").depends_on == ("cleanup",)


def test_rerun_from_stage_invalidates_downstream_outputs_only():
    graph = WorkflowGraph.for_generation_request(
        workflow_id="wf-1",
        mode="Text to 3D",
        text_prompt="asset",
        image_path=None,
        texture_model="texture-provider",
    ).with_completed_generation(output_path="asset.glb")

    rerun = graph.rerun_from("mesh")

    assert rerun.stage("prompt").status is WorkflowStageStatus.SUCCEEDED
    assert rerun.stage("selected_image").status is WorkflowStageStatus.SUCCEEDED
    assert rerun.stage("mesh").status is WorkflowStageStatus.READY
    assert rerun.stage("cleanup").status is WorkflowStageStatus.WAITING
    assert rerun.stage("texture").status is WorkflowStageStatus.WAITING
    assert rerun.stage("texture").artifacts == ()


def test_rerun_from_bundled_texture_restarts_at_mesh():
    graph = WorkflowGraph.for_generation_request(
        workflow_id="wf-1",
        mode="Text to 3D",
        text_prompt="asset",
        image_path=None,
        texture_model="texture-provider",
    ).with_completed_generation(output_path="asset.glb")

    rerun = graph.rerun_from("texture")

    assert graph.rerun_start_key("texture") == "mesh"
    assert graph.rerun_notice("texture") == "Texture is bundled with Mesh; rerun restarts at Mesh."
    assert rerun.stage("prompt").status is WorkflowStageStatus.SUCCEEDED
    assert rerun.stage("mesh").status is WorkflowStageStatus.READY
    assert rerun.stage("cleanup").status is WorkflowStageStatus.WAITING
    assert rerun.stage("texture").status is WorkflowStageStatus.WAITING


def test_completed_generation_maps_manifest_artifacts_to_stages():
    graph = WorkflowGraph.for_generation_request(
        workflow_id="wf-1",
        mode="Image to 3D",
        text_prompt=None,
        image_path="input.png",
        texture_model="texture-provider",
    )

    completed = graph.with_completed_generation(
        output_path="asset_textured.glb",
        run_artifacts={
            "input_images": ["input.png"],
            "generated_images": [],
            "mesh_paths": ["asset_raw.glb", "asset_textured.glb"],
            "texture_paths": ["metadata/texture.png"],
            "final_model_path": "asset_textured.glb",
        },
    )

    assert completed.stage("selected_image").artifacts == ("input.png",)
    assert completed.stage("mesh").artifacts == ("asset_raw.glb", "asset_textured.glb")
    assert completed.stage("cleanup").artifacts == ("asset_raw.glb", "asset_textured.glb")
    assert completed.stage("texture").status is WorkflowStageStatus.SUCCEEDED
    assert completed.stage("texture").artifacts == (
        "metadata/texture.png",
        "asset_textured.glb",
    )
    assert completed.stage("export").status is WorkflowStageStatus.READY
    assert completed.artifact_paths() == (
        "input.png",
        "asset_raw.glb",
        "asset_textured.glb",
        "metadata/texture.png",
    )


def test_workflow_round_trips_through_manifest_dict():
    graph = WorkflowGraph.for_generation_request(
        workflow_id="wf-1",
        mode="Image to 3D",
        image_path="input.png",
        texture_model=None,
    ).with_stage("selected_image", WorkflowStageStatus.SUCCEEDED)

    restored = WorkflowGraph.from_dict(graph.to_dict())

    assert restored == graph


def test_editing_workflow_keeps_rotation_and_inpaint_as_separate_branches():
    graph = WorkflowGraph.for_editing(workflow_id="edit-1", model_path="asset.glb")

    assert [stage.key for stage in graph.stages] == [
        "load_model",
        "rotate",
        "texture_inpaint",
        "export",
    ]
    assert graph.stage("rotate").depends_on == ("load_model",)
    assert graph.stage("texture_inpaint").depends_on == ("load_model",)
    assert graph.stage("export").depends_on == ("rotate", "texture_inpaint")
