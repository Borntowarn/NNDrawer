import { useContext, useState } from 'react'
import { MainContext } from '../../../context/MainContext'
import constants from '../../../constants/constants'
import axios from 'axios'
import './RegistrationModal.css'

export default function RegistrationModal() {
  const {regModalActive, setRegModalActive, setAuthModalActive, setAuth} = useContext(MainContext)
  const [login, setLogin] = useState('')
  const [password, setPassword] = useState('')
  const [username, setUsername] = useState('')
  const [mail, setMail] = useState('')
  
  const handleSubmit = async (e) => {
    e.preventDefault()
    console.log('SUBMIT: ', login, password, username, mail)

    try {
        const response = await axios.post(constants.urls.reg, 
          JSON.stringify({
              login: login,
              password: password,
              username: mail,
              mail: mail,
          }),
          {
            headers: {
              "Accept": "application/json",
              "Content-Type": "application/json",
            },
          }
        )
        
        const id = response.data.user.id
        setAuth(id)
        setRegModalActive(false)
        console.log('USER: ', mail, ' WAS REGISTERED')
    } catch(err) {
        console.log("ERROR: ", err)
    }
  }

  const handleBackRef = () => {
    setRegModalActive(false)
    setAuthModalActive(true)
  }

  return (
    <div className={regModalActive ? 'reg-modal active' : 'reg-modal'}>
      <div className='reg-modal-content'>
        <div className='reg-title'>Enter your data</div>
        <form className='reg-form' onSubmit={handleSubmit}>
          <div className='input-block'>
            <label className='input-label'>Login</label>
            <input className='reg-input' onChange={(e) => setLogin(e.target.value)} value={login} type="text" placeholder='Enter your login'/>
          </div>
          <div className='input-block'>
            <label className='input-label'>Password</label>
            <input className='reg-input' onChange={(e) => setPassword(e.target.value)} value={password} type="password" placeholder='Password'/>
          </div>
          <div className='input-block'>
            <label className="input-label">UserName</label>
            <input className='reg-input' onChange={(e) => setUsername(e.target.value)} value={username} type="text" placeholder='Enter your username'/>
          </div>
          <div className='input-block'>
            <label className="input-label">Email</label>
            <input className='reg-input' onChange={(e) => setMail(e.target.value)} value={mail} type="text" placeholder='Enter your mail'/>
          </div>
          <button className='reg-button' type="submit">send</button>
        </form>
        <div className='reg-sub-buttons'>
            <a className='back-ref' onClick={() => handleBackRef()}>back to authorization</a>
        </div>
      </div>
    </div>
  )
}
