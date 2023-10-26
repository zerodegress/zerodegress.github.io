import { Box, Grid, Typography } from "@suid/material"

export function Contact() {
  return (
    <Grid container flexGrow={1} justifyContent='center'>
      <Grid xs={10}>
        <Box marginBottom='2rem'>
          <Typography variant='h4'>
            QQ
          </Typography>
          <Typography>
            934321107
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