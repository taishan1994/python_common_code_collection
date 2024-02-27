
# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Spider-Syn: Spider Syn Dataset for evaluating Text-SQL models"""
import glob
import json
import os
from turtle import down
import datasets
from typing import List, Generator, Any, Dict, Tuple

logger = datasets.logging.get_logger(__name__)

_CITATION = """
测试
"""

_DESCRIPTION = """\
    测试datasets加载数据集
"""

_HOMEPAGE = "https://zenodo.org/record/5205322#.Yh-B1uhByUl"

_LICENSE = "CC BY-SA 4.0"

_URL = "测试"


class MyDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="测试数据加载",
            version=VERSION,
            description="测试数据",
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs) -> None:
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()
        self.include_train_others: bool = kwargs.pop("include_train_others", False)

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "input": datasets.Value("string"),
                "output": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        # downloaded_filepath = dl_manager.download_and_extract(url_or_urls=_URL)
        downloaded_filepath = './data'

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "train.json")],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "test.json")],
                },
            ),
        ]

    def _generate_examples(
        self, data_filepaths: List[str]) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
        """This function returns the examples in the raw (text) form."""
        for data_filepath in data_filepaths:
            print(data_filepath)
            logger.info("generating examples from = %s", data_filepath)
            with open(data_filepath, encoding="utf-8") as f:
                spider = json.load(f)
                for idx, sample in enumerate(spider):
                    yield idx, {
                        "input": sample["input"],
                        "output": sample["output"],
                    }

