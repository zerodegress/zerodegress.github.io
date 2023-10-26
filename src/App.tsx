import './App.less'
import { Routes, Route } from '@solidjs/router'
import Root from './routes/Root'
import Projects from './routes/Projects'
import About from './routes/About'
import Contact from './routes/Contact'

function App() {
  return (
    <Routes>
      <Route path='/' component={Root}>
        <Route path='about' component={About} />
        <Route path='/projects' component={Projects} />
        <Route path='/contact' component={Contact} />
      </Route>
    </Routes>
  )
}

export default App
