
import random
import sycret
import numpy as np

def sampleBits(seed,expand_len) -> int:
    if seed is None:
        # random.seed(time.time())
        random.seed(123)
    else:
        random.seed(seed)
    return random.getrandbits(expand_len)

class GroupElement(object):
    def __init__(self, value, bitlen,repr_value=None):
        assert (bitlen >= 1), "Improper bit length or scale"
       
        self.bitlen = bitlen
        self.Modulo = 2 ** self.bitlen
       
        if repr_value is None:
            self.value = (int(value) + 2 ** self.bitlen) % (2 ** self.bitlen)
        else:
            self.value = repr_value

    @classmethod
    def unpack(cls,binary,bitlen):
        value = int.from_bytes(binary,"big")
        return GroupElement(value,bitlen)
    
    def getNegVal(self):
        return 2 ** self.bitlen - self.value


    def __add__(self, other):
        assert (type(other) is GroupElement), "Non groupType"
        assert (other.bitlen == self.bitlen), "can only be applied in the same bit length"

        value = (self.value + other.value) & (self.Modulo - 1)
        return GroupElement(value=None, bitlen=self.bitlen, repr_value=value)

    def __sub__(self, other):
        assert (type(other) is GroupElement), "Non groupType"
        assert (other.bitlen == self.bitlen), "can only be applied in the same bit length"

        value = (self.value - other.value+ self.Modulo) & (self.Modulo - 1)
        return GroupElement(value=None, bitlen=self.bitlen, repr_value=value)

    def __gt__(self, other):
        assert (type(other) is GroupElement), "Non groupType"
        assert (other.bitlen == self.bitlen), "can only be applied in the same bit length"
        return self.value > other.value
    
    def __lt__(self, other):
        assert (type(other) is GroupElement), "Non groupType"
        assert (other.bitlen == self.bitlen), "can only be applied in the same bit length"
        return self.value < other.value

    def __eq__(self, other):
        assert (type(other) is GroupElement), "Non groupType"
        assert (other.bitlen == self.bitlen), "can only be applied in the same bit length"
        return self.value == other.value

    def __getitem__(self, item):
        assert (self.bitlen >= item >= 0), f"No index at {item}"
        return self.value >> (self.bitlen-1-item) & 1
    
    def selfPrint(self):
        print("val is: ",self.value)
    
    def ele2Str(self):
        tmp=""
        for i in range(self.getLen()):
            tmp += str(self[i])
        return tmp
    
    def getLen(self):
        return self.bitlen

    def getValue(self):
        return self.value
    
    def packData(self):
        byteLen = int((self.bitlen+7)/8)
        # print("byteLen is: ",byteLen)
        return bytearray( self.value.to_bytes(byteLen,'big') )
    

class NewICKey(object):
    """
    cw_payload0: a arithmetical secret sharing of beta
    cw_payload1: correction word
    dcfKey
    """
    def __init__(self):
        self.CW = 0
        self.dcf_key = None
   
class ICNew:
    """
    Interval Containment Test: 
        If evaluation input x is \in [0,N/2] return b1, otherwise return b2. (b1,b2 are output group elements)
        (Notice, if b2 is not given, then b2 by default is equal to zero).
    """
    def __init__(self,sec_para=128,ring_len=1):
        self.sec_para = sec_para
        self.ring_len = ring_len
        self.dcf = sycret.LeFactory(n_threads=6)
    
    def keyGen(self,inputLen=32):
        # DCF keys
        dcfk0, dcfk1 = self.dcf.keygen(1)
        alpha_a = np.frombuffer(np.ascontiguousarray(
            dcfk0[:, 0:4]), dtype=np.uint32)
        alpha_b = np.frombuffer(np.ascontiguousarray(
            dcfk1[:, 0:4]), dtype=np.uint32)
        gamma = alpha_a + alpha_b
        
        r_in = GroupElement( gamma - (2**32-1), 32)
        r_in0 = GroupElement(sampleBits(None,inputLen) , inputLen)
        r_in1 = r_in - r_in0

        #Calculate the second correction word
        alpha_p = r_in
        alpha_q = r_in + GroupElement( 1<<(inputLen-1), inputLen)
        alpha_q_prime = alpha_q + GroupElement( 1, inputLen) 

        scale = 0
        scale += 1 if alpha_p > alpha_q else 0
        scale -= 1 if alpha_p.getValue() > 0 else 0
        scale += 1 if alpha_q_prime.getValue() > ( (1<<(inputLen-1)) +1 ) else 0
        scale += 1 if alpha_q.getValue() == ( (1<<inputLen)-1 ) else 0
        cw_payload = GroupElement(scale%2, self.ring_len)

        k0 = NewICKey()
        k0.CW = GroupElement(sampleBits(None, self.ring_len) , self.ring_len)
        k0.dcf_key = dcfk0

        k1 = NewICKey()
        k1.CW = cw_payload - k0.CW
        k1.dcf_key = dcfk1

        return r_in0.getValue(),r_in1.getValue(),k0,k1
    
    # Start the online evaluation phase
    """
    param:: zeta is an masked integer value;
    param:: key is an ICNew key
    """
    def eval(self,_id, zeta, key):
        inputLen = 32
        zeta = GroupElement(zeta, 32)

        scale = 1 if zeta.getValue() > 0 else 0
        scale -= 1 if zeta.getValue() > ( (1<<(inputLen-1)) +1 ) else 0
        scale *= _id
        
        out = GroupElement(scale, self.ring_len)

        x_p = zeta + GroupElement( (1<<inputLen)-1, inputLen)
        x_q_prime = zeta + GroupElement( (1<<(inputLen-1))-2, inputLen)

        x_pConvert = np.array( [np.int64(x_p.getValue())] )
        x_q_primeConvert = np.array( [np.int64(x_q_prime.getValue())] )

        eval0 = GroupElement( self.dcf.eval(_id,x_pConvert,key.dcf_key)[0].item(), self.ring_len) 
        eval1 = GroupElement(self.dcf.eval(_id,x_q_primeConvert,key.dcf_key)[0].item(), self.ring_len)

        out -= eval0
        out += eval1
        out += key.CW
        
        return out.getValue()

if __name__=="__main__":
    inputLen = 32

    ic = ICNew()
    r0,r1,k0,k1 = ic.keyGen()

    zeta = 1
    zeta += r0
    zeta += r1

    zeta%=2**32
    
    v0 = ic.eval(0,zeta,k0)
    v1 = ic.eval(1,zeta,k1)

    ret = (v0+v1)%2

    print("test result is: ", 1-v0 )
