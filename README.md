SudoKu-solution
===

Introduction
---
受到Mr Chu的启发，开发了基于python的一个数独solver以及一个相对简陋的WEB服务。这里首先感谢[prajwalkr](https://github.com/prajwalkr)和[tfeldmann](https://github.com/tfeldmann)的sudoku代码：[https://github.com/prajwalkr/SnapSudoku](https://github.com/prajwalkr/SnapSudoku)，
[https://github.com/tfeldmann/Sudoku](https://github.com/prajwalkr/SnapSudoku).


Usage
---
可以在我的[Sudoku主页](http://www.dwwd.fun:8000/index)上下载或者在网站上在线处理，but remind，在线处理的功能仅仅是demo版本，由于算法原因还有很大的改进空间，，而提供下载的软件的效果要相对好一些。
网页
---
![Alt text](https://github.com/yeiamx/SketchToImg-WEB/raw/master/Screenshots/index.png)</br>
点击Download按钮下载文件，速度可能会有点慢，不过文件本身不大也就是了。点击Process按钮进入在线处理。
</br>
![Alt text](https://github.com/yeiamx/SketchToImg-WEB/raw/master/Screenshots/process.png)</br>
点击Choose按钮选择本地图片，选择完毕后将为你计算数独并返回答案（If it could）。
![Alt text](https://github.com/yeiamx/SketchToImg-WEB/raw/master/Screenshots/process_result.png)</br>
网站会播放一首背景音乐，不想听的话静音便可。

软件
---
为了兼容不同环境，已经将python文件打包成了exe文件。解压sudoku.zip，进入sudoku文件夹打开sudoku.exe.它会调用摄像头，将数独图片置于摄像头前，它将为你检测数独并显示解（If it could).点击q键退出程序。


