import unittest
from unittest import TestCase

from e2cnn.nn import *
from e2cnn.gspaces import *

import torch

import random


batchnormalizations = [
    ([('regular_bnorm', 'pointwise')], InnerBatchNorm),
    ([('g_bnorm', 'norm')], GNormBatchNorm),
    ([('norm_bnorm', 'norm')], NormBatchNorm),
    ([('indnorm_bnorm', 'induced_norm')], InducedNormBatchNorm),
]
allbatchnormalizations = []
for bn, _ in batchnormalizations:
    allbatchnormalizations += bn

poolings = [
    ([('regular_mpool', 'pointwise')], PointwiseMaxPool),
    ([('norm_mpool', 'norm')], NormMaxPool),
]

allpoolings = []
for pl, _ in poolings:
    allpoolings += pl

nonlinearities = [
    ([('p_relu', 'pointwise')], PointwiseNonLinearity),
    ([('p_sigmoid', 'pointwise')], PointwiseNonLinearity),
    ([('p_tanh', 'pointwise')], PointwiseNonLinearity),
    ([('c_relu', 'concatenated')], ConcatenatedNonLinearity),
    ([('c_sigmoid', 'concatenated')], ConcatenatedNonLinearity),
    ([('c_tanh', 'concatenated')], ConcatenatedNonLinearity),
    ([('n_relu', 'norm')], NormNonLinearity),
    ([('n_sigmoid', 'norm')], NormNonLinearity),
    ([('vectorfield', 'vectorfield')], VectorFieldNonLinearity),
    ([('gate', 'gate'), ('gated', 'gated')], GatedNonLinearity2),
]

allnonlinearities = []
for nl, _ in nonlinearities:
    allnonlinearities += nl

convolutions = [
    ([('conv2d', 'any')], R2Conv),
]

allconvolutions = []
for cl, _ in convolutions:
    allconvolutions += cl

allfunctions = allbatchnormalizations + allpoolings + allnonlinearities + allconvolutions


class TestNonLinearitiesRotations(TestCase):
    
    def test_cyclic_multiples_nonlinearities_sorted(self):
        N = 8
        g = Rot2dOnR2(N)
        
        reprs = []
        labels = []
        modules = []
        
        gated = 0
        
        for blocks, module in nonlinearities:
            for name, type in blocks:
                if name != 'gate':
                    for r in g.representations.values():
                        if type in r.supported_nonlinearities:
                            reprs.append(r)
                            labels.append(name)
                            
                            if name == 'gated':
                                gated += 1

        reprs = [g.trivial_repr] * gated + reprs
        labels = ['gate'] * gated + labels
        
        r = FieldType(g, reprs)
        
        reprs_dict = r.group_by_labels(labels)
        
        for blocks, module in nonlinearities:
            repr = tuple(reprs_dict[l] for l, _ in blocks)
            if len(repr) == 1:
                repr = repr[0]
            lbs = [l for l, _ in blocks]
            if len(lbs) == 1:
                lbs = lbs[0]
            modules.append((module(repr, function=blocks[0][0]), lbs))
        
        nnl = MultipleModule(r, labels, modules, reshuffle=False)
        nnl.check_equivariance(full_space_action=False)
    
    def test_cyclic_multiples_poolings_sorted(self):
        N = 8
        g = Rot2dOnR2(N)
        
        reprs = []
        labels = []
        modules = []
        
        kernel = (3, 3)
        
        for blocks, module in poolings:
            # print(blocks)
            for name, type in blocks:
                for r in g.representations.values():
                    if type in r.supported_nonlinearities:
                        reprs.append(r)
                        labels.append(name)
        
        r = FieldType(g, reprs)
        
        reprs_dict = r.group_by_labels(labels)
        
        for blocks, module in poolings:
            repr = tuple(reprs_dict[l] for l, _ in blocks)
            if len(repr) == 1:
                repr = repr[0]
            lbs = [l for l, _ in blocks]
            if len(lbs) == 1:
                lbs = lbs[0]
            modules.append((module(repr, kernel_size=kernel), lbs))
        
        nnl = MultipleModule(r, labels, modules, reshuffle=False)
        
        nnl.check_equivariance(full_space_action=False)
    
    def test_cyclic_multiples_batchnorm_sorted(self):
        N = 8
        g = Rot2dOnR2(N)
        
        M = N // 2
        for m in range(M // 2 + 1):
            g.induced_repr(M, g.fibergroup.subgroup(M)[0].irrep(m))

        reprs = []
        labels = []
        modules = []

        for blocks, module in batchnormalizations:
            if module not in [NormBatchNorm, InducedNormBatchNorm]:
                for name, type in blocks:
                    for r in g.representations.values():
                        if type in r.supported_nonlinearities:
                            reprs.append(r)
                            labels.append(name)

        for r in g.representations.values():
            if not r.contains_trivial():
                for blocks, module in batchnormalizations:
                    if module == NormBatchNorm:
                        for name, type in blocks:
                            if type in r.supported_nonlinearities:
                                reprs.append(r)
                                labels.append(name)
            
                    elif module == InducedNormBatchNorm:
                        for name, type in blocks:
                            if any(snl.startswith(type) for snl in r.supported_nonlinearities):
                                reprs.append(r)
                                labels.append(name)

        r = FieldType(g, reprs)

        reprs_dict = r.group_by_labels(labels)
        
        for blocks, module in batchnormalizations:
            repr = tuple(reprs_dict[l] for l, _ in blocks)
            if len(repr) == 1:
                repr = repr[0]
            lbs = [l for l, _ in blocks]
            if len(lbs) == 1:
                lbs = lbs[0]
            modules.append((module(repr), lbs))
        
        nnl = MultipleModule(r, labels, modules, reshuffle=False)
        
        nnl.train()
        b, c, h, w = 4, r.size, 30, 30
        for i in range(20):
            x = GeometricTensor(torch.randn(b, c, h, w), r)
            nnl(x)
        
        nnl.eval()
        
        nnl.check_equivariance(full_space_action=False)
    
    def test_cyclic_multiples_nonlinearities_shuffled(self):
        N = 8
        g = Rot2dOnR2(N)
        
        reprs = []
        labels = []
        modules = []
        
        gated = 0
        
        for blocks, module in nonlinearities:
            # print(blocks)
            for name, type in blocks:
                if name != 'gate':
                    for r in g.representations.values():
                        if type in r.supported_nonlinearities:
                            reprs.append(r)
                            labels.append(name)
                            
                            if name == 'gated':
                                gated += 1

        reprs = [g.trivial_repr] * gated + reprs
        labels = ['gate'] * gated + labels
        
        t = list(zip(reprs, labels))
        
        random.shuffle(t)
        
        reprs, labels = zip(*t)
        
        r = FieldType(g, reprs)
        
        reprs_dict = r.group_by_labels(labels)
        
        for blocks, module in nonlinearities:
            repr = tuple(reprs_dict[l] for l, _ in blocks)
            if len(repr) == 1:
                repr = repr[0]
            lbs = [l for l, _ in blocks]
            if len(lbs) == 1:
                lbs = lbs[0]
            modules.append((module(repr, function=blocks[0][0]), lbs))
        
        nnl = MultipleModule(r, labels, modules, reshuffle=False)
        nnl.check_equivariance(full_space_action=False)
    
    def test_cyclic_multiples_poolings_shuffled(self):
        N = 8
        g = Rot2dOnR2(N)
        
        reprs = []
        labels = []
        modules = []
        
        kernel = (3, 3)
        
        for blocks, module in poolings:
            # print(blocks)
            for name, type in blocks:
                for r in g.representations.values():
                    if type in r.supported_nonlinearities:
                        reprs.append(r)
                        labels.append(name)
        
        t = list(zip(reprs, labels))
        
        random.shuffle(t)
        
        reprs, labels = zip(*t)
        
        r = FieldType(g, reprs)
        
        reprs_dict = r.group_by_labels(labels)
        
        for blocks, module in poolings:
            repr = tuple(reprs_dict[l] for l, _ in blocks)
            if len(repr) == 1:
                repr = repr[0]
            lbs = [l for l, _ in blocks]
            if len(lbs) == 1:
                lbs = lbs[0]
            modules.append((module(repr, kernel_size=kernel), lbs))
        
        nnl = MultipleModule(r, labels, modules, reshuffle=False)
        
        nnl.check_equivariance(full_space_action=False)
    
    def test_cyclic_multiples_batchnorm_shuffled(self):
        N = 8
        g = Rot2dOnR2(N)
        
        M =N//2
        for m in range(M//2+1):
            g.induced_repr(M, g.fibergroup.subgroup(M)[0].irrep(m))

        reprs = []
        labels = []
        modules = []
        
        for blocks, module in batchnormalizations:
            if module not in [NormBatchNorm, InducedNormBatchNorm]:
                for name, type in blocks:
                    for r in g.representations.values():
                        if type in r.supported_nonlinearities:
                            reprs.append(r)
                            labels.append(name)
                            
        for r in g.representations.values():
            if not r.contains_trivial():
                for blocks, module in batchnormalizations:
                    if module == NormBatchNorm:
                        for name, type in blocks:
                            if type in r.supported_nonlinearities:
                                reprs.append(r)
                                labels.append(name)
                            
                    elif module == InducedNormBatchNorm:
                        for name, type in blocks:
                            if any(snl.startswith(type) for snl in r.supported_nonlinearities):
                                reprs.append(r)
                                labels.append(name)

        t = list(zip(reprs, labels))
        
        # for r, l in t:
        #     print(r, l, r.contains_trivial(), r.is_trivial())

        random.shuffle(t)
        
        reprs, labels = zip(*t)
        
        r = FieldType(g, reprs)
        
        reprs_dict = r.group_by_labels(labels)
        
        for blocks, module in batchnormalizations:
            repr = tuple(reprs_dict[l] for l, _ in blocks)
            if len(repr) == 1:
                repr = repr[0]
            lbs = [l for l, _ in blocks]
            if len(lbs) == 1:
                lbs = lbs[0]
            modules.append((module(repr), lbs))
        
        nnl = MultipleModule(r, labels, modules, reshuffle=False)
        
        nnl.train()
        b, c, h, w = 4, r.size, 30, 30
        for i in range(20):
            x = GeometricTensor(torch.randn(b, c, h, w), r)
            nnl(x)
        
        nnl.eval()
        
        nnl.check_equivariance(full_space_action=False)
    
    def test_cyclic_multiples_nonlinearities_sort(self):
        N = 8
        g = Rot2dOnR2(N)
        
        reprs = []
        labels = []
        modules = []
        
        gated = 0
        
        for blocks, module in nonlinearities:
            # print(blocks)
            for i in range(3):
                for name, type in blocks:
                    if name != 'gate':
                        for r in g.representations.values():
                            if type in r.supported_nonlinearities:
                                reprs.append(r)
                                labels.append(name)
                                
                                if name == 'gated':
                                    gated += 1

        reprs = [g.trivial_repr] * gated + reprs
        labels = ['gate'] * gated + labels

        t = list(zip(reprs, labels))
        
        random.shuffle(t)
        
        reprs, labels = zip(*t)
        
        r = FieldType(g, reprs)
        
        reprs_dict = r.group_by_labels(labels)
        
        for blocks, module in nonlinearities:
            repr = tuple(reprs_dict[l] for l, _ in blocks)
            if len(repr) == 1:
                repr = repr[0]
            
            lbs = [l for l, _ in blocks]
            if len(lbs) == 1:
                lbs = lbs[0]
            
            modules.append((module(repr, function=blocks[0][0]), lbs))
        
        nnl = MultipleModule(r, labels, modules, reshuffle=True)
        nnl.check_equivariance(full_space_action=False)
    
    def test_cyclic_multiples_poolings_sort(self):
        N = 8
        g = Rot2dOnR2(N)
        
        reprs = []
        labels = []
        modules = []
        
        kernel = (3, 3)
        
        for blocks, module in poolings:
            # print(blocks)
            for name, type in blocks:
                for r in g.representations.values():
                    if type in r.supported_nonlinearities:
                        reprs.append(r)
                        labels.append(name)
        
        t = list(zip(reprs, labels))
        
        random.shuffle(t)
        
        reprs, labels = zip(*t)
        
        r = FieldType(g, reprs)
        
        reprs_dict = r.group_by_labels(labels)
        
        for blocks, module in poolings:
            repr = tuple(reprs_dict[l] for l, _ in blocks)
            if len(repr) == 1:
                repr = repr[0]
            lbs = [l for l, _ in blocks]
            if len(lbs) == 1:
                lbs = lbs[0]
            modules.append((module(repr, kernel_size=kernel), lbs))
        
        nnl = MultipleModule(r, labels, modules, reshuffle=True)
        
        nnl.check_equivariance(full_space_action=False)
    
    def test_cyclic_multiples_batchnorm_sort(self):
        N = 8
        g = Rot2dOnR2(N)

        M = N // 2
        for m in range(M // 2 + 1):
            g.induced_repr(M, g.fibergroup.subgroup(M)[0].irrep(m))

        reprs = []
        labels = []
        modules = []

        for blocks, module in batchnormalizations:
            if module not in [NormBatchNorm, InducedNormBatchNorm]:
                for name, type in blocks:
                    for r in g.representations.values():
                        if type in r.supported_nonlinearities:
                            reprs.append(r)
                            labels.append(name)

        for r in g.representations.values():
            if not r.contains_trivial():
                for blocks, module in batchnormalizations:
                    if module == NormBatchNorm:
                        for name, type in blocks:
                            if type in r.supported_nonlinearities:
                                reprs.append(r)
                                labels.append(name)
            
                    elif module == InducedNormBatchNorm:
                        for name, type in blocks:
                            if any(snl.startswith(type) for snl in r.supported_nonlinearities):
                                reprs.append(r)
                                labels.append(name)

        t = list(zip(reprs, labels))

        random.shuffle(t)
        
        reprs, labels = zip(*t)
        
        r = FieldType(g, reprs)
        
        reprs_dict = r.group_by_labels(labels)
        
        for blocks, module in batchnormalizations:
            repr = tuple(reprs_dict[l] for l, _ in blocks)
            if len(repr) == 1:
                repr = repr[0]
            lbs = [l for l, _ in blocks]
            if len(lbs) == 1:
                lbs = lbs[0]
            modules.append((module(repr), lbs))
        
        nnl = MultipleModule(r, labels, modules, reshuffle=True)
        
        nnl.train()
        b, c, h, w = 4, r.size, 30, 30
        for i in range(20):
            x = GeometricTensor(torch.randn(b, c, h, w), r)
            nnl(x)
        
        nnl.eval()
        
        nnl.check_equivariance(full_space_action=False)

    def test_so2_multiples_nonlinearities_sorted(self):
        N = 8
        g = Rot2dOnR2(-1, N)
    
        reprs = []
        labels = []
        modules = []
    
        gated = 0
        
        for blocks, module in nonlinearities:
            # print(blocks)
            for name, type in blocks:
                if name != 'gate':
                    for r in g.representations.values():
                        if type in r.supported_nonlinearities:
                            
                            reprs.append(r)
                            labels.append(name)
                        
                            if name == 'gated':
                                gated += 1

        reprs = [g.trivial_repr] * gated + reprs
        labels = ['gate'] * gated + labels
    
        r = FieldType(g, reprs)
    
        reprs_dict = r.group_by_labels(labels)
    
        for blocks, module in nonlinearities:
            if all(l in reprs_dict for l, _ in blocks):
                repr = tuple(reprs_dict[l] for l, _ in blocks)
                if len(repr) == 1:
                    repr = repr[0]
                lbs = [l for l, _ in blocks]
                if len(lbs) == 1:
                    lbs = lbs[0]
                modules.append((module(repr, function=blocks[0][0]), lbs))
    
        nnl = MultipleModule(r, labels, modules, reshuffle=False)
        nnl.check_equivariance(full_space_action=False)

    def test_so2_multiples_poolings_sorted(self):
        N = 8
        g = Rot2dOnR2(-1, N)
    
        reprs = []
        labels = []
        modules = []
    
        kernel = (3, 3)
    
        for blocks, module in poolings:
            # print(blocks)
            for name, type in blocks:
                for r in g.representations.values():
                    if type in r.supported_nonlinearities:
                        reprs.append(r)
                        labels.append(name)
    
        r = FieldType(g, reprs)
    
        reprs_dict = r.group_by_labels(labels)
    
        for blocks, module in poolings:
            if all(l in reprs_dict for l, _ in blocks):
                repr = tuple(reprs_dict[l] for l, _ in blocks)
                if len(repr) == 1:
                    repr = repr[0]
                lbs = [l for l, _ in blocks]
                if len(lbs) == 1:
                    lbs = lbs[0]
                modules.append((module(repr, kernel_size=kernel), lbs))
    
        nnl = MultipleModule(r, labels, modules, reshuffle=False)
    
        nnl.check_equivariance(full_space_action=False)

    def test_so2_multiples_batchnorm_sorted(self):
        N = 8
        g = Rot2dOnR2(-1, N)

        reprs = []
        labels = []
        modules = []

        for blocks, module in batchnormalizations:
            if module not in [NormBatchNorm, InducedNormBatchNorm]:
                for name, type in blocks:
                    for r in g.representations.values():
                        if type in r.supported_nonlinearities:
                            reprs.append(r)
                            labels.append(name)

        for r in g.representations.values():
            if not r.contains_trivial():
                for blocks, module in batchnormalizations:
                    if module == NormBatchNorm:
                        for name, type in blocks:
                            if type in r.supported_nonlinearities:
                                reprs.append(r)
                                labels.append(name)
            
        r = FieldType(g, reprs)

        reprs_dict = r.group_by_labels(labels)
    
        for blocks, module in batchnormalizations:
            if all(l in reprs_dict for l, _ in blocks):
                repr = tuple(reprs_dict[l] for l, _ in blocks)
                if len(repr) == 1:
                    repr = repr[0]
                lbs = [l for l, _ in blocks]
                if len(lbs) == 1:
                    lbs = lbs[0]
                modules.append((module(repr), lbs))
    
        nnl = MultipleModule(r, labels, modules, reshuffle=False)
    
        nnl.train()
        b, c, h, w = 4, r.size, 30, 30
        for i in range(20):
            x = GeometricTensor(torch.randn(b, c, h, w), r)
            nnl(x)
    
        nnl.eval()
    
        nnl.check_equivariance(full_space_action=False)

    def test_so2_multiples_nonlinearities_shuffled(self):
        N = 8
        g = Rot2dOnR2(-1, N)
    
        reprs = []
        labels = []
        modules = []
    
        gated = 0
    
        for blocks, module in nonlinearities:
            # print(blocks)
            for name, type in blocks:
                if name != 'gate':
                    for r in g.representations.values():
                        if type in r.supported_nonlinearities:
                            reprs.append(r)
                            labels.append(name)
                        
                            if name == 'gated':
                                gated += 1

        reprs = [g.trivial_repr] * gated + reprs
        labels = ['gate'] * gated + labels
    
        t = list(zip(reprs, labels))
    
        random.shuffle(t)
    
        reprs, labels = zip(*t)
    
        r = FieldType(g, reprs)
    
        reprs_dict = r.group_by_labels(labels)
    
        for blocks, module in nonlinearities:
            if all(l in reprs_dict for l, _ in blocks):
                repr = tuple(reprs_dict[l] for l, _ in blocks)
                if len(repr) == 1:
                    repr = repr[0]
                lbs = [l for l, _ in blocks]
                if len(lbs) == 1:
                    lbs = lbs[0]
                modules.append((module(repr, function=blocks[0][0]), lbs))
    
        nnl = MultipleModule(r, labels, modules, reshuffle=False)
        nnl.check_equivariance(full_space_action=False)

    def test_so2_multiples_poolings_shuffled(self):
        N = 8
        g = Rot2dOnR2(-1, N)
    
        reprs = []
        labels = []
        modules = []
    
        kernel = (3, 3)
    
        for blocks, module in poolings:
            # print(blocks)
            for name, type in blocks:
                for r in g.representations.values():
                    if type in r.supported_nonlinearities:
                        reprs.append(r)
                        labels.append(name)
    
        t = list(zip(reprs, labels))
    
        random.shuffle(t)
    
        reprs, labels = zip(*t)
    
        r = FieldType(g, reprs)
    
        reprs_dict = r.group_by_labels(labels)
    
        for blocks, module in poolings:
            if all(l in reprs_dict for l, _ in blocks):
                repr = tuple(reprs_dict[l] for l, _ in blocks)
                if len(repr) == 1:
                    repr = repr[0]
                lbs = [l for l, _ in blocks]
                if len(lbs) == 1:
                    lbs = lbs[0]
                modules.append((module(repr, kernel_size=kernel), lbs))
    
        nnl = MultipleModule(r, labels, modules, reshuffle=False)
    
        nnl.check_equivariance(full_space_action=False)

    def test_so2_multiples_batchnorm_shuffled(self):
        N = 8
        g = Rot2dOnR2(-1, N)

        reprs = []
        labels = []
        modules = []

        for blocks, module in batchnormalizations:
            if module not in [NormBatchNorm, InducedNormBatchNorm]:
                for name, type in blocks:
                    for r in g.representations.values():
                        if type in r.supported_nonlinearities:
                            reprs.append(r)
                            labels.append(name)

        for r in g.representations.values():
            if not r.contains_trivial():
                for blocks, module in batchnormalizations:
                    if module == NormBatchNorm:
                        for name, type in blocks:
                            if type in r.supported_nonlinearities:
                                reprs.append(r)
                                labels.append(name)

        t = list(zip(reprs, labels))

        random.shuffle(t)
    
        reprs, labels = zip(*t)
    
        r = FieldType(g, reprs)
    
        reprs_dict = r.group_by_labels(labels)
    
        for blocks, module in batchnormalizations:
            if all(l in reprs_dict for l, _ in blocks):
                repr = tuple(reprs_dict[l] for l, _ in blocks)
                if len(repr) == 1:
                    repr = repr[0]
                lbs = [l for l, _ in blocks]
                if len(lbs) == 1:
                    lbs = lbs[0]
                modules.append((module(repr), lbs))
    
        nnl = MultipleModule(r, labels, modules, reshuffle=False)
    
        nnl.train()
        b, c, h, w = 4, r.size, 30, 30
        for i in range(20):
            x = GeometricTensor(torch.randn(b, c, h, w), r)
            nnl(x)
    
        nnl.eval()
    
        nnl.check_equivariance(full_space_action=False)

    def test_so2_multiples_nonlinearities_sort(self):
        N = 8
        g = Rot2dOnR2(-1, N)
    
        reprs = []
        labels = []
        modules = []
    
        gated = 0
    
        for blocks, module in nonlinearities:
            # print(blocks)
            for i in range(3):
                for name, type in blocks:
                    if name != 'gate':
                        for r in g.representations.values():
                            if type in r.supported_nonlinearities:
                                reprs.append(r)
                                labels.append(name)
                            
                                if name == 'gated':
                                    gated += 1

        reprs = [g.trivial_repr] * gated + reprs
        labels = ['gate'] * gated + labels
    
        t = list(zip(reprs, labels))
    
        random.shuffle(t)
    
        reprs, labels = zip(*t)
    
        r = FieldType(g, reprs)
    
        reprs_dict = r.group_by_labels(labels)
    
        for blocks, module in nonlinearities:
            if all(l in reprs_dict for l, _ in blocks):
                repr = tuple(reprs_dict[l] for l, _ in blocks)
                if len(repr) == 1:
                    repr = repr[0]
            
                lbs = [l for l, _ in blocks]
                if len(lbs) == 1:
                    lbs = lbs[0]
            
                modules.append((module(repr, function=blocks[0][0]), lbs))
    
        nnl = MultipleModule(r, labels, modules, reshuffle=True)
        nnl.check_equivariance(full_space_action=False)

    def test_so2_multiples_poolings_sort(self):
        N = 8
        g = Rot2dOnR2(-1, N)
    
        reprs = []
        labels = []
        modules = []
    
        kernel = (3, 3)
    
        for blocks, module in poolings:
            # print(blocks)
            for name, type in blocks:
                for r in g.representations.values():
                    if type in r.supported_nonlinearities:
                        reprs.append(r)
                        labels.append(name)
    
        t = list(zip(reprs, labels))
    
        random.shuffle(t)
    
        reprs, labels = zip(*t)
    
        r = FieldType(g, reprs)
    
        reprs_dict = r.group_by_labels(labels)
    
        for blocks, module in poolings:
            if all(l in reprs_dict for l, _ in blocks):
                repr = tuple(reprs_dict[l] for l, _ in blocks)
                if len(repr) == 1:
                    repr = repr[0]
                lbs = [l for l, _ in blocks]
                if len(lbs) == 1:
                    lbs = lbs[0]
                modules.append((module(repr, kernel_size=kernel), lbs))
    
        nnl = MultipleModule(r, labels, modules, reshuffle=True)
    
        nnl.check_equivariance(full_space_action=False)

    def test_so2_multiples_batchnorm_sort(self):
        N = 8
        g = Rot2dOnR2(-1, N)

        reprs = []
        labels = []
        modules = []

        for blocks, module in batchnormalizations:
            if module not in [NormBatchNorm, InducedNormBatchNorm]:
                for name, type in blocks:
                    for r in g.representations.values():
                        if type in r.supported_nonlinearities:
                            reprs.append(r)
                            labels.append(name)

        for r in g.representations.values():
            if not r.contains_trivial():
                for blocks, module in batchnormalizations:
                    if module == NormBatchNorm:
                        for name, type in blocks:
                            if type in r.supported_nonlinearities:
                                reprs.append(r)
                                labels.append(name)
            

        t = list(zip(reprs, labels))

        random.shuffle(t)
    
        reprs, labels = zip(*t)
    
        r = FieldType(g, reprs)
    
        reprs_dict = r.group_by_labels(labels)
    
        for blocks, module in batchnormalizations:
            if all(l in reprs_dict for l, _ in blocks):
                repr = tuple(reprs_dict[l] for l, _ in blocks)
                if len(repr) == 1:
                    repr = repr[0]
                lbs = [l for l, _ in blocks]
                if len(lbs) == 1:
                    lbs = lbs[0]
                modules.append((module(repr), lbs))
    
        nnl = MultipleModule(r, labels, modules, reshuffle=True)
    
        nnl.train()
        b, c, h, w = 4, r.size, 30, 30
        for i in range(20):
            x = GeometricTensor(torch.randn(b, c, h, w), r)
            nnl(x)
    
        nnl.eval()
    
        nnl.check_equivariance(full_space_action=False)


if __name__ == '__main__':
    unittest.main()
