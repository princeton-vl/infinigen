import bpy

import numpy as np


    """
    An enum for all node types.
    Pass these as the first argument to nw.new_node(...)

    Organized according to the Blender UI's Shift-A 'Add Node' Menu .
    If the different node contexts disagree, then we tiebreak by using geonodes' placement, not shader nodes
    """

    # Attribute
    Attribute = "ShaderNodeAttribute"
    CaptureAttribute = "GeometryNodeCaptureAttribute"
    AttributeStatistic = 'GeometryNodeAttributeStatistic'
    TransferAttribute = "GeometryNodeAttributeTransfer"
    StoreNamedAttribute = "GeometryNodeStoreNamedAttribute"

    # Color Menu
    ColorRamp = "ShaderNodeValToRGB"
    MixRGB = "ShaderNodeMixRGB"
    RGBCurve = "ShaderNodeRGBCurve"
    BrightContrast = "CompositorNodeBrightContrast"
    Exposure = 'CompositorNodeExposure'
    CombineHSV = 'ShaderNodeCombineHSV'
    SeparateRGB = 'ShaderNodeSeparateRGB'
    CombineRGB = 'ShaderNodeCombineRGB'

    # Curve
    CurveToMesh = "GeometryNodeCurveToMesh"
    MeshToCurve = "GeometryNodeMeshToCurve"
    SampleCurve = 'GeometryNodeSampleCurve'
    SetCurveRadius = 'GeometryNodeSetCurveRadius'
    CurveLength = 'GeometryNodeCurveLength'
    SplineParameter = 'GeometryNodeSplineParameter'
    SubdivideCurve = 'GeometryNodeSubdivideCurve'
    ResampleCurve = 'GeometryNodeResampleCurve'
    TrimCurve = 'GeometryNodeTrimCurve'
    SplineLength = 'GeometryNodeSplineLength'

    # Curve Primitves
    QuadraticBezier = 'GeometryNodeCurveQuadraticBezier'
    CurveCircle = 'GeometryNodeCurvePrimitiveCircle'
    BezierSegment = "GeometryNodeCurvePrimitiveBezierSegment"

    # Geometry
    SetPosition = "GeometryNodeSetPosition"
    JoinGeometry = "GeometryNodeJoinGeometry"
    MergeByDistance = "GeometryNodeMergeByDistance"
    SeparateGeometry = "GeometryNodeSeparateGeometry"
    BoundingBox = "GeometryNodeBoundBox"
    Transform = "GeometryNodeTransform"
    Raycast = 'GeometryNodeRaycast'
    DuplicateElements = 'GeometryNodeDuplicateElements'

    # Input
    GroupInput = "NodeGroupInput"
    RGB = "ShaderNodeRGB"
    Value = "ShaderNodeValue"
    RandomValue = "FunctionNodeRandomValue"
    CollectionInfo = "GeometryNodeCollectionInfo"
    ObjectInfo = "GeometryNodeObjectInfo"
    ObjectInfo_Shader = 'ShaderNodeObjectInfo'
    Vector = "FunctionNodeInputVector"
    InputPosition = "GeometryNodeInputPosition"
    InputNormal = "GeometryNodeInputNormal"
    InputColor = "FunctionNodeInputColor"
    TextureCoord = "ShaderNodeTexCoord"
    Index = 'GeometryNodeInputIndex'
    AmbientOcclusion = 'ShaderNodeAmbientOcclusion'
    NamedAttribute = 'GeometryNodeInputNamedAttribute'

    # Instances
    RealizeInstances = "GeometryNodeRealizeInstances"
    InstanceOnPoints = "GeometryNodeInstanceOnPoints"
    TranslateInstances = "GeometryNodeTranslateInstances"
    RotateInstances = "GeometryNodeRotateInstances"
    ScaleInstances = "GeometryNodeScaleInstances"

    # Material
    SetMaterial = "GeometryNodeSetMaterial"

    # Mesh
    SubdivideMesh = "GeometryNodeSubdivideMesh"
    SubdivisionSurface = "GeometryNodeSubdivisionSurface"
    MeshToPoints = 'GeometryNodeMeshToPoints'
    SetShadeSmooth = 'GeometryNodeSetShadeSmooth'
    ExtrudeMesh = 'GeometryNodeExtrudeMesh'
    FlipFaces = 'GeometryNodeFlipFaces'

    # Mesh Primitives
    MeshCircle = "GeometryNodeMeshCircle"
    MeshGrid = 'GeometryNodeMeshGrid'
    MeshLine = 'GeometryNodeMeshLine'
    MeshUVSphere = 'GeometryNodeMeshUVSphere'
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
    Clamp = "ShaderNodeClamp"

    # Texture
    NoiseTexture = "ShaderNodeTexNoise"
    MusgraveTexture = "ShaderNodeTexMusgrave"
    VoronoiTexture = "ShaderNodeTexVoronoi"
    WaveTexture = "ShaderNodeTexWave"
    WhiteNoiseTexture = 'ShaderNodeTexWhiteNoise'
    # Shaders
    MixShader = "ShaderNodeMixShader"
    PrincipledBSDF = "ShaderNodeBsdfPrincipled"
    PrincipledVolume = "ShaderNodeVolumePrincipled"
    PrincipledHairBSDF = 'ShaderNodeBsdfHairPrincipled'

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

    Nodes.VectorMath: ['operation'],
    Nodes.BooleanMath: ['operation'],
    Nodes.Compare: ['mode', 'data_type', 'operation'],
    Nodes.VoronoiTexture: ['voronoi_dimensions', 'feature', 'distance'],

    Nodes.Attribute: ['attribute_name', 'attribute_type'],
    Nodes.CaptureAttribute: ['domain', 'data_type'],
    Nodes.TextureCoord: ['from_instancer'],
    Nodes.PrincipledBSDF: ['distribution', 'subsurface_method'],

    Nodes.Mapping: ['vector_type'],
    Nodes.MapRange: ['data_type', 'interpolation_type', 'clamp'],
    Nodes.MixRGB: ['blend_type'],


    Nodes.RandomValue: ['data_type'],

    Nodes.SeparateGeometry: ['domain'],
    Nodes.MergeByDistance: ['mode'],

    Nodes.MeshBoolean: ['operation'],
    Nodes.MeshCircle: ['fill_type'],
    Nodes.NamedAttribute: ['data_type'],
    Nodes.StoreNamedAttribute: ['data_type', 'domain'],
    Nodes.ResampleCurve: ['mode'],
    Nodes.TrimCurve: ['mode'],
    Nodes.MeshLine: ['mode'],
    Nodes.MeshToPoints: ['mode'],

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
    Nodes.SeparateColor: ['mode'],
}

# Certain nodes should only be created once. This list defines which ones.

# Map the type of a socket (ie, .outputs[0].type), to the corresponding value to put into a 
# data_type attr, ie CaptureAttributes data_type. Frustratingly these are not directly related. 
NODETYPE_TO_DATATYPE = {
    'FLOAT_COLOR': 'RGBA',

NODECLASS_TO_DATATYPE = {
    'NodeSocketVector': 'FLOAT_VECTOR',

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
    bpy.types.Scene: Nodes.Composite,
    bpy.types.World: Nodes.WorldOutput,
    bpy.types.NodesModifier: Nodes.GroupOutput,
    bpy.types.ShaderNodeGroup: Nodes.GroupOutput,
    bpy.types.CompositorNodeGroup: Nodes.GroupOutput,
}

DATATYPE_FIELDS = {
    'FLOAT': 'value',
    'INT': 'value',
    'FLOAT_VECTOR': 'vector',
