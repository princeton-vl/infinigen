Exporters
=========

Public entrypoints for rendering scenes and exporting generated data.

Rendering
---------

.. autofunction:: infinigen2.exporters.render_cycles.render_cycles

.. autofunction:: infinigen2.exporters.render_cycles.render_cycles_ground_truth

.. autofunction:: infinigen2.exporters.render_eevee.render_eevee

.. autofunction:: infinigen2.exporters.render_eevee.render_eevee_ground_truth

.. autofunction:: infinigen2.exporters.render_workbench.render_workbench

Data export
-----------

.. autofunction:: infinigen2.exporters.imu.save_imu

.. autofunction:: infinigen2.exporters.object_data.save_object_data

.. autofunction:: infinigen2.exporters.rigidbody_point_tracks.save_point_tracks

Ground-truth visualization
--------------------------

.. autofunction:: infinigen2.exporters.visualize_gt.visualize_gt

.. autofunction:: infinigen2.exporters.visualize_gt.visualize_depth

.. autofunction:: infinigen2.exporters.visualize_gt.visualize_flow

.. autofunction:: infinigen2.exporters.visualize_gt.visualize_normals

.. autofunction:: infinigen2.exporters.visualize_gt.visualize_seg_mask

.. autofunction:: infinigen2.exporters.visualize_gt.visualize_uniq_inst

.. autofunction:: infinigen2.exporters.visualize_gt.visualize_bw

.. autofunction:: infinigen2.exporters.visualize_gt.visualize_object_boxes

Validation
----------

.. autofunction:: infinigen2.exporters.render_error_check.render_validity_check

.. autofunction:: infinigen2.exporters.render_error_check.unsafe_displacement_materials

.. autofunction:: infinigen2.exporters.render_error_check.check_material_uv_coords

.. autofunction:: infinigen2.exporters.render_error_check.count_material_nodes

.. autofunction:: infinigen2.exporters.render_error_check.normal_input_used

.. autofunction:: infinigen2.exporters.render_error_check.unlinked_texture_vector

.. autofunction:: infinigen2.exporters.render_error_check.missing_attribute_issues

.. autofunction:: infinigen2.exporters.render_error_check.nonfinite_vertex_counts

.. autofunction:: infinigen2.exporters.render_error_check.singular_transform_objects

.. autofunction:: infinigen2.exporters.render_error_check.hidden_render_objects

.. autofunction:: infinigen2.exporters.render_error_check.detect_cycles_errors

.. autofunction:: infinigen2.exporters.render_error_check.backfacing.check_backfacing

Types
-----

.. autoclass:: infinigen2.exporters.DenoiseMode
   :members:

.. autoclass:: infinigen2.exporters.RenderEeveeParams
   :members:

.. autoclass:: infinigen2.exporters.RenderPass
   :members:

.. autoclass:: infinigen2.exporters.ExportType
   :members:

.. autoclass:: infinigen2.exporters.DisplacementMode
   :members:
