import { Outlet, useNavigate } from "@solidjs/router"
import { Box,Button,Grid,Typography } from "@suid/material"

export function Root() {
  const navigate = useNavigate()
  return (
    <Box flexGrow={1}>
      <Grid container justifyContent='center'>
        <Grid xs={24} marginBottom='2rem'>
          <Typography variant='h1' textAlign='center'>
            冰点启航
          </Typography>
          <Box sx={{ 
            width: '100%',
            display: 'flex',
            justifyContent: 'center',
          }}>
            <Button onClick={() => navigate('/about')}>
              关于
            </Button>
            <Button onClick={() => navigate('/projects')}>
              项目
            </Button>
            <Button onClick={() => navigate('/contact')}>
              联系方式
            </Button>
          </Box>
        </Grid>
        <Grid xs={24}>
          <Outlet />
        </Grid>
      </Grid>
    </Box>
  )
}

export default Root