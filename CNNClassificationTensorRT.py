# ---------------------------------------------------------
# CNNClassificationTensorRT
# Copyright (c) 2019
# Licensed under The MIT License [see LICENSE for details]
# Written by Rudy Nurhadi
# ---------------------------------------------------------

import os
import cv2
import time
import json
import numpy as np
import tensorrt as trt

import pycuda.autoinit
import pycuda.driver as cuda

from threading import Thread


class CNNClassificationTensorRT():
    def __init__(self, model_dir, model_name, max_batch_size, rebuild_engine=False):
        self.MODEL_NAME = model_name

        self.CWD_PATH = model_dir

        self.PATH_TO_MODEL = os.path.join(self.CWD_PATH, '%s.uff' % self.MODEL_NAME)
        self.PATH_TO_ENGINE = os.path.join(self.CWD_PATH, '%s.engine' % self.MODEL_NAME)
        self.PATH_TO_META = os.path.join(self.CWD_PATH, '%s.meta' % self.MODEL_NAME)

        with open(self.PATH_TO_META, 'r') as fp:
            self.meta = json.load(fp)

        self.MAX_BATCH_SIZE = max_batch_size
        self.MAX_WORKSPACE_SIZE = 1 << 29

        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.DTYPE = trt.float16

        # Model
        self.INPUT_NAME = self.meta['inp_name']
        self.INPUT_SHAPE = self.meta['inp_size']
        self.OUTPUT_NAME = self.meta['out_name']
        
        self.prepare_engine(rebuild_engine)
        
    def allocate_buffers(self, engine):
        print('allocate buffers')
        
        h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), trt.nptype(engine.get_binding_dtype(0)))
        h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), trt.nptype(engine.get_binding_dtype(1)))
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        
        stream = cuda.Stream()
        
        return stream, h_input, d_input, h_output, d_output


    def build_engine(self, model_file):
        print('build engine...')
        
        with trt.Builder(self.TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
            builder.max_workspace_size = self.MAX_WORKSPACE_SIZE
            builder.max_batch_size = self.MAX_BATCH_SIZE
            if self.DTYPE == trt.float16:
                builder.fp16_mode = True
                builder.strict_type_constraints = True
                print("using float16 precision")
            parser.register_input(self.INPUT_NAME, self.INPUT_SHAPE, trt.UffInputOrder.NHWC)
            parser.register_output(self.OUTPUT_NAME)
            parser.parse(model_file, network, self.DTYPE)
            
            return builder.build_cuda_engine(network)


    def load_input(self, img, host_buffer):
        img_array = cv2.resize(img, (self.INPUT_SHAPE[1], self.INPUT_SHAPE[0])).astype(trt.nptype(self.DTYPE)).ravel()
        img_array = img_array / 255.
        np.copyto(host_buffer, img_array)


    def do_inference(self, context, stream, h_input, d_input, h_output, d_output):
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input, h_input, stream)

        # Run inference.
        context.execute_async(batch_size=1, bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        
        return h_output

    def prepare_engine(self, rebuild_engine):
        self.engine = {}
        self.stream = {}
        self.h_input = {}
        self.d_input = {}
        self.h_output = {}
        self.d_output = {}
        self.context = {}
        self.output = {}
        
        print("prepare engine")
        try:
            if rebuild_engine:
                raise("rebuild engine")
            for i in range(self.MAX_BATCH_SIZE):
                with open(self.PATH_TO_ENGINE, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
                    self.engine[i] = runtime.deserialize_cuda_engine(f.read())
        except:
            engine_tmp = self.build_engine(self.PATH_TO_MODEL)
            with open(self.PATH_TO_ENGINE, "wb") as f:
                f.write(engine_tmp.serialize())
            del engine_tmp
            for i in range(self.MAX_BATCH_SIZE):
                with open(self.PATH_TO_ENGINE, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
                    self.engine[i] = runtime.deserialize_cuda_engine(f.read())
        
        for i in range(self.MAX_BATCH_SIZE):
            self.stream[i], self.h_input[i], self.d_input[i], self.h_output[i], self.d_output[i] = self.allocate_buffers(self.engine[i])
            self.context[i] = self.engine[i].create_execution_context()
            
        print("engine ready")
        
    def return_predict(self, imgs):
        results = []
        imgs_new = []
        imgs_sub = []
        num_loop = int((len(imgs) - 1) / self.MAX_BATCH_SIZE) + 1
        
        for i in range(len(imgs)):
            if i > 0 and i % self.MAX_BATCH_SIZE == 0:
                imgs_new.append(imgs_sub)
                imgs_sub = []
            imgs_sub.append(imgs[i])
            
        imgs_new.append(imgs_sub)
        imgs = imgs_new
        
        for n in range(num_loop):
            batch_size = len(imgs[n])
            loadInputThreads = {}
            for i in range(batch_size):
                #self.load_input(imgs[n][i], self.h_input[i])
                loadInputThreads[i] = Thread(target=self.load_input,
                                             args=(imgs[n][i], self.h_input[i],))
                loadInputThreads[i].start()
            
            for i in range(batch_size):
                loadInputThreads[i].join()
                self.output[i] = self.do_inference(self.context[i], self.stream[i], self.h_input[i], self.d_input[i], self.h_output[i], self.d_output[i])
                
            for i in range(batch_size):
                self.stream[i].synchronize()
                output = self.output[i]
                results.append({"labels": self.meta["labels"], "confidences": output})
        return results
