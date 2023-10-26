import { Box, Grid, List, ListItem, Typography } from '@suid/material'

export function Projects() {
  return (
    <Grid container flexGrow={1} justifyContent='center'>
      <Grid xs={10}>
        <Box marginBottom='2rem'>
          <Typography variant='h3' fontWeight='bold'>
            个人项目
          </Typography>
          <Typography variant='h6'>
            大多在GitHub上发布，
            <a href='https://github.com/zerodegress'>就在这里</a>。
            不必惊讶，半成品就是这么多，毕竟我想写就写，开发比较任性。
          </Typography>
        </Box>
        <Box marginBottom='2rem'>
          <Typography variant='h3' fontWeight='bold'>
            参与项目
          </Typography>
          <Typography variant='h6'>能提的上号的大概就以下几个：</Typography>
          <List>
            <ListItem>
              <Box>
                <Box>
                  <a href='https://github.com/RW-HPS/RW-HPS'>
                    <Typography variant='h4' fontWeight='bold'>
                      RW-HPS
                    </Typography>
                  </a>
                </Box>
                <Box>
                  <Typography>
                    给一个不咋知名的RTS游戏Rusted
                    Warfare（中文名一般叫做“铁锈战争”）开发的游戏服务器软件，
                    服务器支持插件扩展其功能，我开发了其中JS插件支持部分（不算烂，但也不算什么好活，
                    基本啥基础功能都没提供，就是单纯的把JS脚本通过graaljs当成java注入进去），
                    并编写了一个
                    <a href='https://github.com/RW-HPS/RW-HPS-JSLib'>
                      简单的框架
                    </a>
                    （没太完成，抽空重构一下，顺便补充补充文档）。
                  </Typography>
                </Box>
              </Box>
            </ListItem>
            <ListItem>
              <Box>
                <Box>
                  <a href='https://github.com/CFPAOrg/Minecraft-Mod-Language-Package'>
                    <Typography variant='h4' fontWeight='bold'>
                      Minecraft I18N 汉化资源包
                    </Typography>
                  </a>
                </Box>
                <Box>
                  <Typography>
                    偶尔发现玩Minecraft用的模组没汉化会到这里做下汉化PR。
                  </Typography>
                </Box>
              </Box>
            </ListItem>
          </List>
        </Box>
      </Grid>
    </Grid>
  )
}

export default Projects
