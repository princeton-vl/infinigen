DEVICE_FUNC float smooth_union(float d1, float d2, float k) {
    float h = max(k - abs(d1 - d2), 0.0f);
    return min(d1, d2) - h*h*0.25/ k;
}

DEVICE_FUNC float smooth_subtraction(float d1, float d2, float k)
{
    return -smooth_union(d2, -d1, k);
}
