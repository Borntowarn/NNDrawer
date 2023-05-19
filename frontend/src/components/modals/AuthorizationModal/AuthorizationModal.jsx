import { useContext, useState } from 'react'
import { MainContext } from '../../../context/MainContext'
import axios from 'axios'
import constants from '../../../constants/constants'
import './AuthorizationModal.css'

export default function AuthorizationModal() {
  const { authModalActive, setAuthModalActive, setRegModalActive, setAuth, createNewProject, setProjects } = useContext(MainContext)
  const [login, setLogin] = useState('')
  const [password, setPassword] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    console.log('SUBMIT: ', login, password)
    try {
        const response = await axios.post(constants.urls.auth, 
          JSON.stringify({
              login: login,
              password: password 
          }),
          {
            headers: {
              "Accept": "application/json",
              "Content-Type": "application/json",
            },
          }
        )
        
        const id = response.data.user.id
        const mail = response.data.user.mail

        setAuth(id)
        setAuthModalActive(false)
        const importProjects =  response.data.projects.map((elem) => {
          return JSON.parse(elem.data)
        })
        setProjects(importProjects)

        localStorage.setItem('user-token', response.data.token);

        console.log('USER: ', mail, ' WAS AUTHORIZED')
    } catch(err) {
        console.log("ERROR: ", err)
    }
  }

  const handleTestProject = () => {
    createNewProject()
    setAuthModalActive(false)
  }

  const handleRegForm = () => {
    setAuthModalActive(false)
    setRegModalActive(true)
  }

  return (
    <div className={authModalActive ? 'modal active' : 'modal'}>
        <div className='modal-content'>
            <div className='auth-title'>Welcom to NNdrawer!</div>
            <form className='auth-form' onSubmit={handleSubmit}>
                <label htmlFor="auth-log" className='input-label'>Login</label>
                <input id='auth-log' className='auth-input' onChange={(e) => setLogin(e.target.value)} value={login} type="text" placeholder='Login'/>
                <label htmlFor="auth-pass" className='input-label'>Password</label>
                <input id='auth-pass' className='auth-input' onChange={(e) => setPassword(e.target.value)} value={password} type="password" placeholder='Password'/>
                <button className='auth-button' type="submit">send</button>
            </form>
            <div className='sub-buttons'>
                <a className='reg-ref' onClick={() => handleRegForm()}>registration</a>
                <a className='test-ref' onClick={() => handleTestProject()}>create test project</a>
            </div>
        </div>
    </div>
  )
}
