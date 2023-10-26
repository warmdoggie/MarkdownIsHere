# Markdown运行中报错提示
### 2023-10-26
### 1.报错command 'markdown.showPreviewToSide' not found

    问题：无法预览，Markdown插件更新后与VSC不兼容
    解决方法：在插件中下载旧版本重启即可
### 2.报错 command 'markdown.extension.onBackspaceKey' not found
问题：热键被占用
解决方法：
1.在问题1解决后也没有热键占用的问题了
2.文件->首选项->键盘快捷方式->搜索"backspace"重新修改一个热键
    
```
    "key": "shift+backspace",//任意修改
    "command": "markdown.extension.onBackspaceKey"
```