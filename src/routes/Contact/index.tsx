import { Box, Grid, Typography } from "@suid/material"

export function Contact() {
  return (
    <Grid container flexGrow={1} justifyContent='center'>
      <Grid xs={10}>
        <Box marginBottom='2rem'>
          <Typography variant='h4'>
            KOOK群
          </Typography>
          <Typography>
            <a href='https://kook.top/8iuCok'>https://kook.top/8iuCok</a>
          </Typography>
        </Box>
        <Box>
          <Typography variant='h4'>
            邮箱
          </Typography>
          <Typography>
            zerodegress@outlook.com
          </Typography>
        </Box>
      </Grid>
    </Grid>
  )
}

export default Contact