# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: all infinigen authors


import bpy

import numpy as np


class Nodes:
    """
    An enum for all node types.
    Pass these as the first argument to nw.new_node(...)

    Organized according to the Blender UI's Shift-A 'Add Node' Menu .
    If the different node contexts disagree, then we tiebreak by using geonodes' placement, not shader nodes
    """

    Mix = "ShaderNodeMix"

    # Attribute
    Attribute = "ShaderNodeAttribute"
    CaptureAttribute = "GeometryNodeCaptureAttribute"
    AttributeStatistic = 'GeometryNodeAttributeStatistic'
    TransferAttribute = "GeometryNodeAttributeTransfer" # removed in b3.4, still supported via compatibility.py
    DomainSize = 'GeometryNodeAttributeDomainSize'
    StoreNamedAttribute = "GeometryNodeStoreNamedAttribute"
    NamedAttribute = 'GeometryNodeInputNamedAttribute'
    SampleIndex = "GeometryNodeSampleIndex"
    SampleNearest = "GeometryNodeSampleNearest"
    SampleNearestSurface = "GeometryNodeSampleNearestSurface"


    # Color Menu
    ColorRamp = "ShaderNodeValToRGB"
    MixRGB = "ShaderNodeMixRGB"
    RGBCurve = "ShaderNodeRGBCurve"
    BrightContrast = "CompositorNodeBrightContrast"
    Exposure = 'CompositorNodeExposure'
    CombineHSV = 'ShaderNodeCombineHSV'
    SeparateRGB = 'ShaderNodeSeparateRGB'
    SeparateColor = 'ShaderNodeSeparateColor'
    CombineRGB = 'ShaderNodeCombineRGB'
    CombineColor = 'ShaderNodeCombineColor'

    #bl3.5 additions
    SeparateComponents = 'GeometryNodeSeparateComponents'
    SetID = 'GeometryNodeSetID'
    InterpolateCurves = 'GeometryNodeInterpolateCurves'
    SampleUVSurface = 'GeometryNodeSampleUVSurface'
    MeshIsland = 'GeometryNodeInputMeshIsland'
    IsViewport = 'GeometryNodeIsViewport'
    ImageInfo = 'GeometryNodeImageInfo'
    CurveofPoint = 'GeometryNodeCurveOfPoint'
    CurvesInfo = 'ShaderNodeHairInfo'
    Radius = 'GeometryNodeInputRadius'
    EvaluateonDomain = 'GeometryNodeFieldOnDomain'
    BlurAttribute = 'GeometryNodeBlurAttribute'
    EndpointSelection = 'GeometryNodeCurveEndpointSelection'
    PointsofCurve = 'GeometryNodePointsOfCurve'
    SetSplineResolution = 'GeometryNodeSetSplineResolution'
    OffsetPointinCurve = 'GeometryNodeOffsetPointInCurve'
    SplineResolution = 'GeometryNodeInputSplineResolution'

    # Curve
    CurveToMesh = "GeometryNodeCurveToMesh"
    CurveToPoints = "GeometryNodeCurveToPoints"
    MeshToCurve = "GeometryNodeMeshToCurve"
    SampleCurve = 'GeometryNodeSampleCurve'
    SetCurveRadius = 'GeometryNodeSetCurveRadius'
    SetCurveTilt = 'GeometryNodeSetCurveTilt'
    CurveLength = 'GeometryNodeCurveLength'
    CurveSplineType = 'GeometryNodeCurveSplineType'
    SetHandlePositions = 'GeometryNodeSetCurveHandlePositions'
    SetHandleType = 'GeometryNodeCurveSetHandles'
    CurveTangent = 'GeometryNodeInputTangent'
    SplineParameter = 'GeometryNodeSplineParameter'
    SplineType = 'GeometryNodeCurveSplineType'
    SubdivideCurve = 'GeometryNodeSubdivideCurve'
    ResampleCurve = 'GeometryNodeResampleCurve'
    TrimCurve = 'GeometryNodeTrimCurve'
    ReverseCurve = 'GeometryNodeReverseCurve'
    SplineLength = 'GeometryNodeSplineLength'
    FillCurve = 'GeometryNodeFillCurve'

    # Curve Primitves
    QuadraticBezier = 'GeometryNodeCurveQuadraticBezier'
    CurveCircle = 'GeometryNodeCurvePrimitiveCircle'
    CurveLine = 'GeometryNodeCurvePrimitiveLine'
    CurveBezierSegment = "GeometryNodeCurvePrimitiveBezierSegment"
    BezierSegment = "GeometryNodeCurvePrimitiveBezierSegment"

    # Geometry
    SetPosition = "GeometryNodeSetPosition"
    JoinGeometry = "GeometryNodeJoinGeometry"
    MergeByDistance = "GeometryNodeMergeByDistance"
    SeparateGeometry = "GeometryNodeSeparateGeometry"
    BoundingBox = "GeometryNodeBoundBox"
    Transform = "GeometryNodeTransform"
    DeleteGeometry = 'GeometryNodeDeleteGeometry'
    Proximity = "GeometryNodeProximity"
    ConvexHull = "GeometryNodeConvexHull"
    Raycast = 'GeometryNodeRaycast'
    DuplicateElements = 'GeometryNodeDuplicateElements'

    # Input
    GroupInput = "NodeGroupInput"
    RGB = "ShaderNodeRGB"
    Boolean = "FunctionNodeInputBool"
    Value = "ShaderNodeValue"
    RandomValue = "FunctionNodeRandomValue"
    CollectionInfo = "GeometryNodeCollectionInfo"
    ObjectInfo = "GeometryNodeObjectInfo"
    ObjectInfo_Shader = 'ShaderNodeObjectInfo'
    Vector = "FunctionNodeInputVector"
    InputID = "GeometryNodeInputID"
    InputPosition = "GeometryNodeInputPosition"
    InputNormal = "GeometryNodeInputNormal"
    InputEdgeVertices = "GeometryNodeInputMeshEdgeVertices"
    InputEdgeAngle = "GeometryNodeInputMeshEdgeAngle"
    InputColor = "FunctionNodeInputColor"
    InputMeshFaceArea = 'GeometryNodeInputMeshFaceArea'
    TextureCoord = "ShaderNodeTexCoord"
    Index = 'GeometryNodeInputIndex'
    AmbientOcclusion = 'ShaderNodeAmbientOcclusion'
    Integer = 'FunctionNodeInputInt'
    LightPath = 'ShaderNodeLightPath'
    ShortestEdgePath = 'GeometryNodeInputShortestEdgePaths'

    # Instances
    RealizeInstances = "GeometryNodeRealizeInstances"
    InstanceOnPoints = "GeometryNodeInstanceOnPoints"
    TranslateInstances = "GeometryNodeTranslateInstances"
    RotateInstances = "GeometryNodeRotateInstances"
    ScaleInstances = "GeometryNodeScaleInstances"

    # Material
    SetMaterial = "GeometryNodeSetMaterial"
    SetMaterialIndex = "GeometryNodeSetMaterialIndex"
    MaterialIndex = "GeometryNodeInputMaterialIndex"

    # Mesh
    SubdivideMesh = "GeometryNodeSubdivideMesh"
    SubdivisionSurface = "GeometryNodeSubdivisionSurface"
    MeshToPoints = 'GeometryNodeMeshToPoints'
    MeshBoolean = "GeometryNodeMeshBoolean"
    SetShadeSmooth = 'GeometryNodeSetShadeSmooth'
    DualMesh = 'GeometryNodeDualMesh'
    ScaleElements = 'GeometryNodeScaleElements'
    IcoSphere = 'GeometryNodeMeshIcoSphere'
    ExtrudeMesh = 'GeometryNodeExtrudeMesh'
    FlipFaces = 'GeometryNodeFlipFaces'
    FaceNeighbors = 'GeometryNodeInputMeshFaceNeighbors'
    EdgePathToCurve = 'GeometryNodeEdgePathsToCurves'
    DeleteGeom = 'GeometryNodeDeleteGeometry'
    SplitEdges = 'GeometryNodeSplitEdges'

    # Mesh Primitives
    MeshCircle = "GeometryNodeMeshCircle"
    MeshGrid = 'GeometryNodeMeshGrid'
    MeshLine = 'GeometryNodeMeshLine'
    MeshUVSphere = 'GeometryNodeMeshUVSphere'
    MeshIcoSphere = 'GeometryNodeMeshIcoSphere'
    MeshCube = 'GeometryNodeMeshCube'

    # Output Menu
    GroupOutput = "NodeGroupOutput"
    MaterialOutput = "ShaderNodeOutputMaterial"
    LightOutput = "ShaderNodeOutputLight"
    OutputFile = "CompositorNodeOutputFile"
    WorldOutput = "ShaderNodeOutputWorld"
    Composite = "CompositorNodeComposite"
    Viewer = "CompositorNodeViewer"

    # Point    
    DistributePointsOnFaces = "GeometryNodeDistributePointsOnFaces"
    PointsToVertices = "GeometryNodePointsToVertices"
    PointsToVolume = 'GeometryNodePointsToVolume'
    SetPointRadius = 'GeometryNodeSetPointRadius'

    # Vector
    SeparateXYZ = "ShaderNodeSeparateXYZ"
    CombineXYZ = "ShaderNodeCombineXYZ"
    VectorCurve = "ShaderNodeVectorCurve"
    VectorRotate = "ShaderNodeVectorRotate"
    AlignEulerToVector = "FunctionNodeAlignEulerToVector"

    # Volume
    VolumeToMesh = 'GeometryNodeVolumeToMesh'

    # Math
    VectorMath = "ShaderNodeVectorMath"
    Math = "ShaderNodeMath"
    MapRange = 'ShaderNodeMapRange'
    BooleanMath = "FunctionNodeBooleanMath"
    Compare = "FunctionNodeCompare"
    FloatToInt = "FunctionNodeFloatToInt"
    FieldAtIndex = "GeometryNodeFieldAtIndex"
    AccumulateField = "GeometryNodeAccumulateField"
    Clamp = "ShaderNodeClamp"
    Invert = "ShaderNodeInvert"

    # Texture
    NoiseTexture = "ShaderNodeTexNoise"
    MusgraveTexture = "ShaderNodeTexMusgrave"
    VoronoiTexture = "ShaderNodeTexVoronoi"
    WaveTexture = "ShaderNodeTexWave"
    WhiteNoiseTexture = 'ShaderNodeTexWhiteNoise'
    ImageTexture = "GeometryNodeImageTexture"
    GradientTexture = 'ShaderNodeTexGradient'
    ShaderImageTexture = "ShaderNodeTexImage"

    # Shaders
    MixShader = "ShaderNodeMixShader"
    DiffuseBSDF = "ShaderNodeBsdfDiffuse"
    PrincipledBSDF = "ShaderNodeBsdfPrincipled"
    TranslucentBSDF = "ShaderNodeBsdfTranslucent"
    TransparentBSDF = "ShaderNodeBsdfTransparent"
    PrincipledVolume = "ShaderNodeVolumePrincipled"
    PrincipledHairBSDF = 'ShaderNodeBsdfHairPrincipled'
    Emission = 'ShaderNodeEmission'
    Fresnel = 'ShaderNodeFresnel'
    NewGeometry = 'ShaderNodeNewGeometry'
    RefractionBSDF = "ShaderNodeBsdfRefraction"
    GlassBSDF = "ShaderNodeBsdfGlass"
    GlossyBSDF = "ShaderNodeBsdfGlossy"
    LayerWeight = "ShaderNodeLayerWeight"

    # Layout
    Reroute = "NodeReroute"

    # Utilities
    Mapping = "ShaderNodeMapping"
    FloatCurve = "ShaderNodeFloatCurve"
    RotateEuler = "FunctionNodeRotateEuler"
    Switch = "GeometryNodeSwitch"

    # Compositor - Filter
    RenderLayers = "CompositorNodeRLayers"
    LensDistortion = "CompositorNodeLensdist"
    Glare = 'CompositorNodeGlare'

    # World Nodes
    SkyTexture = "ShaderNodeTexSky"
    Background = "ShaderNodeBackground"

    #bl3.5 additions
    SeparateComponents = 'GeometryNodeSeparateComponents'
    SetID = 'GeometryNodeSetID'
    InterpolateCurves = 'GeometryNodeInterpolateCurves'
    SampleUVSurface = 'GeometryNodeSampleUVSurface'
    MeshIsland = 'GeometryNodeInputMeshIsland'
    IsViewport = 'GeometryNodeIsViewport'
    ImageInfo = 'GeometryNodeImageInfo'
    CurveofPoint = 'GeometryNodeCurveOfPoint'
    CurvesInfo = 'ShaderNodeHairInfo'
    Radius = 'GeometryNodeInputRadius'
    EvaluateonDomain = 'GeometryNodeFieldOnDomain'
    BlurAttribute = 'GeometryNodeBlurAttribute'
    EndpointSelection = 'GeometryNodeCurveEndpointSelection'
    PointsofCurve = 'GeometryNodePointsOfCurve'
    SetSplineResolution = 'GeometryNodeSetSplineResolution'
    OffsetPointinCurve = 'GeometryNodeOffsetPointInCurve'
    SplineResolution = 'GeometryNodeInputSplineResolution'

'''
Blender doesnt have an automatic way of discovering what properties
exist on a node that might need to be set but are NOT in .inputs. This dict
documents what types of properties we might need to set on each type of node

Used in transpiler's create_attrs_dict
'''
NODE_ATTRS_AVAILABLE = {

    Nodes.Math: ['operation', 'use_clamp'],
    Nodes.VectorMath: ['operation'],
    Nodes.BooleanMath: ['operation'],
    Nodes.Compare: ['mode', 'data_type', 'operation'],

    Nodes.NoiseTexture: ['noise_dimensions'],
    Nodes.MusgraveTexture: ['musgrave_dimensions', 'musgrave_type'],
    Nodes.VoronoiTexture: ['voronoi_dimensions', 'feature', 'distance'],
    Nodes.GradientTexture: ['gradient_type'],

    Nodes.RGB: ['color'],
    Nodes.Attribute: ['attribute_name', 'attribute_type'],
    Nodes.AttributeStatistic: ['domain', 'data_type'],
    Nodes.CaptureAttribute: ['domain', 'data_type'],
    Nodes.TextureCoord: ['from_instancer'],

    Nodes.PrincipledBSDF: ['distribution', 'subsurface_method'],

    Nodes.Mapping: ['vector_type'],
    Nodes.MapRange: ['data_type', 'interpolation_type', 'clamp'],
    Nodes.ColorRamp: [],  # Color ramp properties are set in special_case_colorramp, since they are nested
    Nodes.MixRGB: ['blend_type'],
    Nodes.Mix: ['data_type', 'blend_type', 'clamp_result', 'clamp_factor'],
    Nodes.AccumulateField: ['data_type'],
    Nodes.CombineRGB: ['mode'],
    Nodes.CombineColor: ['mode'],
    Nodes.SeparateColor: ['mode'],

    Nodes.DistributePointsOnFaces: ['distribute_method'],
    Nodes.CollectionInfo: ['transform_space'],

    Nodes.RandomValue: ['data_type'],

    Nodes.Switch: ['input_type'],
    Nodes.TransferAttribute: ['data_type', 'mapping'], 
    Nodes.SeparateGeometry: ['domain'],
    Nodes.MergeByDistance: ['mode'],

    Nodes.Integer: ['integer'],
    Nodes.MeshBoolean: ['operation'],
    Nodes.MeshCircle: ['fill_type'],
    Nodes.CurveSplineType: ['spline_type'],
    Nodes.SetHandlePositions: ['mode'],
    Nodes.SetHandleType: ['handle_type', 'mode'],
    Nodes.NamedAttribute: ['data_type'],
    Nodes.StoreNamedAttribute: ['data_type', 'domain'],
    Nodes.CurveToPoints: ['mode'],
    Nodes.FillCurve: ['mode'],

    Nodes.ResampleCurve: ['mode'],
    Nodes.TrimCurve: ['mode'],
    Nodes.MeshLine: ['mode'],
    Nodes.MeshToPoints: ['mode'],

    Nodes.DeleteGeom: ['mode'],
    Nodes.Proximity: ['target_element'],

    Nodes.CurveCircle: ['mode'],
    Nodes.SampleCurve: ['mode'],
    Nodes.BezierSegment: ['mode'],
    Nodes.CurveLine: ['mode'],
    Nodes.ExtrudeMesh: ['mode'],
    Nodes.Raycast: ['data_type', 'mapping'],

    Nodes.AlignEulerToVector: ['axis', 'pivot_axis'],
    Nodes.VectorRotate: ['invert', 'rotation_type'],
    Nodes.RotateEuler: ['space', 'type'],
    Nodes.DuplicateElements: ['domain'],
    Nodes.SeparateRGB: ['mode'],
    Nodes.SeparateColor: ['mode'],

    Nodes.DomainSize: ['component']

}

# Certain nodes should only be created once. This list defines which ones.
SINGLETON_NODES = [Nodes.GroupInput, Nodes.GroupOutput, Nodes.MaterialOutput, Nodes.WorldOutput, Nodes.Viewer,
    Nodes.Composite, Nodes.RenderLayers, Nodes.LightOutput]

# Map the type of a socket (ie, .outputs[0].type), to the corresponding value to put into a 
# data_type attr, ie CaptureAttributes data_type. Frustratingly these are not directly related. 
NODETYPE_TO_DATATYPE = {
    'VALUE': 'FLOAT',
    'INT': 'INT',
    'VECTOR': 'FLOAT_VECTOR',
    'FLOAT_COLOR': 'RGBA',
    'BOOLEAN': 'BOOLEAN'}

NODECLASS_TO_DATATYPE = {
    'NodeSocketFloat': 'FLOAT',
    'NodeSocketInt': 'INT',
    'NodeSocketVector': 'FLOAT_VECTOR',
    'NodeSocketColor': 'RGBA',
    'NodeSocketBool': 'BOOLEAN'}

DATATYPE_TO_NODECLASS = {v: k for k, v in NODECLASS_TO_DATATYPE.items()}
NODECLASSES = [k for k in dir(bpy.types) if 'NodeSocket' in k]

PYTYPE_TO_DATATYPE = {
    int: 'INT', 
    float: 'FLOAT', 
    np.float32: 'FLOAT',
    np.float64: 'FLOAT',
    np.array: 'FLOAT_VECTOR', 
    bool: 'BOOLEAN'
}

# Each thing containing nodes has a different output node id
OUTPUT_NODE_IDS = {
    bpy.types.Material: Nodes.MaterialOutput,
    bpy.types.Scene: Nodes.Composite,
    bpy.types.World: Nodes.WorldOutput,
    bpy.types.NodesModifier: Nodes.GroupOutput,
    bpy.types.GeometryNodeGroup: Nodes.GroupOutput,
    bpy.types.ShaderNodeGroup: Nodes.GroupOutput,
    bpy.types.CompositorNodeGroup: Nodes.GroupOutput,
}

DATATYPE_DIMS = {'FLOAT': 1, 'INT': 1, 'FLOAT_VECTOR': 3, 'FLOAT2': 2, 'FLOAT_COLOR': 4, 'BOOLEAN': 1, }
DATATYPE_FIELDS = {
    'FLOAT': 'value',
    'INT': 'value',
    'FLOAT_VECTOR': 'vector',
    'FLOAT_COLOR': 'color',
    'BOOLEAN': 'boolean', }
