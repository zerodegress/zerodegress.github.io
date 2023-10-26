/* @refresh reload */
import { render } from 'solid-js/web'

import './index.less'
import { Router } from '@solidjs/router'
import App from './App'

const root = document.getElementById('root')

render(() => (
  <Router>
    <App />
  </Router>
), root!)
