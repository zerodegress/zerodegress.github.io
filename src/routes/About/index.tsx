import { Box, Grid, Typography } from '@suid/material'

export function About() {
  return (
    <Grid container flexGrow={1} justifyContent='center'>
      <Grid xs={10}>
        <Box marginBottom='2rem'>
          <Typography variant='h3' textAlign='center'>
            一个编程爱好者，一个游戏爱好者
          </Typography>
        </Box>
        <Box marginBottom='2rem'>
          <Typography variant='h3' fontWeight='bold'>
            简介
          </Typography>
          <Typography variant='h6'>
            编程和游戏爱好者，平时都趴在电脑上敲代码或者打电动，
            也可能在钻研某种刚发现的感兴趣的事物。
            偶尔会在Steam上搜寻新游戏，或者在购物网站上盘点需要采购的数码配件。
            不喜欢集体活动，喜欢独处。比起跟人打交道，更愿意和技术性的东西打交道。
          </Typography>
          <Typography variant='h6'>
            常用的网名“冰点启航”，英文网名“ZeroDegress”（其实就是名字拼错了一直用hehehe）。
          </Typography>
        </Box>
        <Box marginBottom='2rem'>
          <Typography variant='h3' fontWeight='bold'>
            技能
          </Typography>
          <Typography variant='h6'>
            擅长前后端编程，掌握语言包括但不限于Rust、JavaScript、Kotlin、Zig、C#。
            更偏好新锐的现代语言（或许除了Go），但是对于C、C++、Java等传统语言也得心应手。
            对于Unity游戏逆向，Java反编译等也有所了解，乐于学习游戏模组的开发。
          </Typography>
          <Typography variant='h6'>
            对于各种编程工具（VSCode，Git，Docker，Virtual Box）都有所认知，
            目前主力代码编辑器是VSCode。
          </Typography>
          <Typography variant='h6'>
            特别喜欢一些小众的编程环境和框架，例如Tauri，SolidJs，Deno之类的。
          </Typography>
          <Typography variant='h6'>
            对于英文翻译也有不小的兴趣，偶尔会给Minecraft的CFPA汉化项目提PR
          </Typography>
          <Typography variant='h6'>
            游戏水平不能说是平平无奇，只能说是非常差劲，因此很少玩多人游戏，
            即使是玩《战地》系列游戏也总是玩单人模式，更不要说CS之类的了。
            因此比起RTS，FPS，Dotalike更偏好回合制、可以即时暂停的游戏，
            或者是生存模拟经营类较为轻松而且慢节奏的游戏。
          </Typography>
          <Typography variant='h6'>
            上一次写作已经是很长时间之前的事情了，但内心也许一直在为未来的作品寻找灵感和构思。
          </Typography>
        </Box>
        <Box marginBottom='2rem'>
          <Typography variant='h3' fontWeight='bold'>
            大概现状
          </Typography>
          <Typography variant='h6'>
            在一所无聊的学校继续做无聊的学习，偶尔跟实验室的朋友们整整小车机器人。
          </Typography>
        </Box>
        <Box marginBottom='2rem'>
          <Typography variant='h3' fontWeight='bold'>
            展望未来
          </Typography>
          <Typography variant='h6'>
            暂时还没想好，也许应该去码头弄点薯条，这就足够了。
          </Typography>
        </Box>
      </Grid>
    </Grid>
  )
}

export default About
