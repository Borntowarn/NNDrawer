import { useContext } from 'react'
import './AuthorizationModal.css'
import { MainContext } from '../../context/MainContext'

export default function AuthorizationModal() {
  const { modalActive } = useContext(MainContext) 

  return (
    <div className={modalActive ? 'modal' : 'modal active'}>
        <div className='modal-content'>
            Welcom to NNdrawer!
        </div>
    </div>
  )
}
