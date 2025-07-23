# mitoclass

[![License GNU GPL v3.0](https://img.shields.io/pypi/l/mitoclass.svg?color=green)](https://github.com/ImHorPhen/mitoclass/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/mitoclass.svg?color=green)](https://pypi.org/project/mitoclass)
[![Python Version](https://img.shields.io/pypi/pyversions/mitoclass.svg?color=green)](https://python.org)
[![tests](https://github.com/ImHorPhen/mitoclass/workflows/tests/badge.svg)](https://github.com/ImHorPhen/mitoclass/actions)
[![codecov](https://codecov.io/gh/ImHorPhen/mitoclass/branch/main/graph/badge.svg)](https://codecov.io/gh/ImHorPhen/mitoclass)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/mitoclass)](https://napari-hub.org/plugins/mitoclass)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

Mitoclassif is a napari plugin for classifying mitochondrial morphology from microscopy images: it allows preprocessing data, training or using a model, predicting classes (connected, fragmented, intermediate), visualizing overlays and 3D summaries, and managing a prediction history.

----------------------------------

This [napari] plugin was generated with [copier] using the [napari-plugin-template].

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `mitoclass` via [pip]:

```
pip install mitoclass
```

If napari is not already installed, you can install `mitoclass` with napari and Qt via:

```
pip install "mitoclass[all]"
```


To install latest development version :

```
pip install git+https://github.com/ImHorPhen/mitoclass.git
```



## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [GNU GPL v3.0] license,
"mitoclass" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[file an issue]: https://github.com/ImHorPhen/mitoclass/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
