import { useContext, useState } from 'react'
import { MainContext } from '../../context/MainContext'
import axios from 'axios'
import './AuthorizationModal.css'

export default function AuthorizationModal() {
  const { modalActive, setModalActive, setAuth, createNewProject } = useContext(MainContext)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')

  // test authorization 
  const AUTHORIZATION_URL = 'api/login'

  const handleEmail = (value) => {
    setEmail(value)
  }

  const handlePassword = (value) => {
    setPassword(value)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    console.log('SUBMIT: ', email, password)
    try {
        const response = await axios.post(AUTHORIZATION_URL, 
          JSON.stringify({
              login: email,
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

        setAuth({id, mail})
        setModalActive(true)

        console.log('USER: ', mail, ' WAS AUTHORIZED')
    } catch(err) {
        console.log("ERROR: ", err)
    }
  }

  const handleTestProject = () => {
    createNewProject()
    setModalActive(true)
  }

  return (
    <div className={modalActive ? 'modal' : 'modal active'}>
        <div className='modal-content'>
            <div className='auth-title'>Welcom to NNdrawer!</div>
            <form className='auth-form' onSubmit={handleSubmit}>
                <input className='auth-input' onChange={(e) => handleEmail(e.target.value)} value={email} type="text" placeholder='Enter your mail'/>
                <input className='auth-input' onChange={(e) => handlePassword(e.target.value)} value={password} type="password" placeholder='password'/>
                <button className='auth-button' type="submit">send</button>
            </form>
            <div className='sub-buttons'>
                <a className='reg-ref'>registration</a>
                <a className='test-ref' onClick={() => handleTestProject()}>create test project</a>
            </div>
        </div>
    </div>
  )
}
