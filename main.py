#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 17:53:35 2022

@author: anup
"""

import os
import inference
from fastapi import FastAPI
from pydantic import BaseModel


class Item(BaseModel):
    source_path: str


app = FastAPI()


@app.post("/process/")
async def create_item(item: Item):
    item_dict = item.dict()
    if not os.path.exists(item_dict['source_path']):
        item_dict['Result'] = "Error"
        return item_dict
    try:
        inference.get_image_output(item_dict['source_path'])
        item_dict['Result'] = "Success"
        return item_dict
    except:
        item_dict['Result'] = "Error"
        return item_dict
