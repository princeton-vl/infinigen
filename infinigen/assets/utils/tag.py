# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yihan Wang


import os
import bpy
import json
import numpy as np
import infinigen.core.util.blender as butil
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler

class AutoTag():
    tag_dict = {}
    def __init__(self):
        self.tag_dict = {}
    
    def rephrase(self, TagName):
        assert len(TagName) > 0, 'TagName is empty'
        tags = TagName.split('.')
        tags.sort()
        name = tags[0]
        for i in range(1, len(tags)):
            name = name + '.' + tags[i]
        return name

    def trigger_update(self):
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.object.mode_set(mode='OBJECT')

    def get_all_objs(self, obj):
        objs = [obj]
        for obj_child in obj.children:
            objs += self.get_all_objs(obj_child)
        return objs

    def add_attribute(self, obj_name, attr_name, type='FLOAT', domain='POINT', value=1.0, recursive=True):
        root_obj = bpy.data.objects[obj_name]
        # print(domain, attr_name, obj_name, type)
        if recursive == False:
            obj = root_obj
            if obj.type != 'MESH':
                attr = None
            else:
                attr = obj.data.attributes.new(attr_name, type, domain)
                val = [value] * len(attr.data)
                attr.data.foreach_set('value', val) 
        else:  
            objs = self.get_all_objs(root_obj)
            for obj in objs:
                if obj.type != 'MESH':
                    attr = None
                else:
                    attr = obj.data.attributes.new(attr_name, type, domain)
                    val = [value] * len(attr.data)
                    attr.data.foreach_set('value', val)                   
        return attr


    # This function now only supports APPLIED OBJECTS
    # PLEASE KEEP ALL THE GEOMETRY APPLIED BEFORE SCATTERING THEM ON THE TERRAIN
    # PLEASE DO NOT USE BOOLEAN TAGS FOR OTHER USE
    def save_tag(self, path='./MaskTag.json'):
        with open(path, 'w') as f:
            json.dump(self.tag_dict, f)
    
    def load_tag(self, path='./MaskTag.json'):
        with open(path, 'r') as f:
            self.tag_dict = json.load(f)

    def relabel_obj(self, root_obj):

        tag_dict = self.tag_dict
        tag_name = [0] * len(tag_dict)
        for name, tag_id in tag_dict.items():
            tag_name[int(tag_id) - 1] = name

        objs = self.get_all_objs(root_obj)
        for obj in objs:
            if obj.type != 'MESH':
                continue
            
            attr_dict = {}
            n = 0
            tag = None
            for name, attr in obj.data.attributes.items():
                if 'TAG_' in name:
                    n = len(attr.data)
                    val = n * [0]
                    attr.data.foreach_get('value', val)
                    attr_dict[name[4:]] = val

                if name == 'MaskTag':
                    n = len(attr.data)
                    tag = n * [0]
                    attr.data.foreach_get('value', tag)

            for name in attr_dict.keys():
                obj.data.attributes.remove(obj.data.attributes['TAG_' + name])
            
            assert (len(attr_dict) > 0) or (tag is not None), 'No tag for object {}'.format(obj.name)

            MaskTag = [0] * n
            for i in range(n):
                TagName = None
                if (tag is not None) and (tag[i] > 0):
                    TagName = tag_name[tag[i] - 1]
                for name, val in attr_dict.items():
                    if val[i] == True:
                        if TagName is None:
                            TagName = name
                        else:
                            TagName = TagName + '.' + name
                TagName = self.rephrase(TagName)
                TagValue = tag_dict.get(TagName, -1)
                if TagValue == -1:
                    TagValue = len(tag_dict) + 1
                    tag_dict[TagName] = TagValue
                    tag_name.append(TagName)
                MaskTag[i] = TagValue
            
            if tag is None:
                MaskTag_attr = self.add_attribute(obj.name, 'MaskTag', type='INT', value=1, recursive=False)
            else:
                MaskTag_attr = obj.data.attributes['MaskTag']

            MaskTag_attr.data.foreach_set('value', MaskTag)
            self.tag_dict = tag_dict

        return root_obj


tag_system = AutoTag()

def tag_object(obj, name=""):
    if name != "":
        name = 'TAG_' + name
        tag_system.add_attribute(obj.name, name, type='BOOLEAN', value=True)
    tag_system.relabel_obj(obj)


def tag_nodegroup(nw, input_node, name):
    name = 'TAG_' + name
    store_named_attribute = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': input_node, 'Name': name, 'Value': True},
        attrs={'domain': 'POINT', 'data_type': 'BOOLEAN'})
    return store_named_attribute
    