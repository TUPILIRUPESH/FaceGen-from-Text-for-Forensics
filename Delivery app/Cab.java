class Cab extends Vehicle {
    @Override
    public void assign() {
        System.out.println("the delivery has been assigned in cab");
    }

    @Override
    public void dispatch() {
        System.out.println("the delivery has been dispatched in cab ");
    }
}