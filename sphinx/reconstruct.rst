Tracking in 3D: Computer Vision
===============================


Suppose you get a bunch of ``oishi`` features in 2D images, how to get 3D coordinates from them?


Very briefly, you typically use the following code


.. code-block:: python

    import fish_3d as f3
    import fish_track as ft

    clusters_nv = []  # nv = n view

    for i in range(n_views):
        clusters = ft.oishi.get_clusters(
            features_nv[i], kernels_nv[i], angles
        )
        clusters_nv.append(clusters)

    matched_indices, matched_centres, reproj_errors = f3.three_view_cluster_match(
        clusters_nv, cameras, tol_2d, sample_size, depth
    )

The ``matched_centres`` is the 3D coordinates constructed from many 2D features.

But what is happening? What are the meanings of those variables? What is the business happening inside the code?

Here is an detailed explaination.


Synchronised Cameras
++++++++++++++++++++

under construction ...

Epipolar Geometry
+++++++++++++++++

under construction ...

Camera Distortion
+++++++++++++++++

under construction ...

Water refraction
++++++++++++++++

under construction ...

Cost function
+++++++++++++

under construction ...
